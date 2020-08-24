//! Fast(?) AtomicArc using Crossbeam to _coalesce_ and _defer_ reference counting
//! 
//! # Quick start
//! 
//! A single atomic load returns a `GuardedArc` valid for the lifetime of a
//! `crossbeam::epoch::Guard`.
//! 
//! ```
//! use std::sync::atomic::Ordering;
//! use crossbeam::epoch;
//! 
//! let a = carc::AtomicArc::new(1234);
//! let mut b;
//! {
//!     let guard = epoch::pin();
//!     let c = a.load(Ordering::Acquire, &guard); // just an atomic load
//!     b = *c;                                    // the refcount is untouched
//! }                                           
//! assert_eq!(b, 1234);
//! 
//! ```
//! 
//! The caller chooses the memory order of the load, allowing fine-grained
//! control over synchronization.
//! 
//! ```
//! use std::sync::atomic;
//! use std::sync::atomic::Ordering;
//! use crossbeam::epoch;
//! 
//! let a = carc::AtomicArc::new(1234);
//! let mut b;
//! {
//!     let guard = epoch::pin();
//!     let c = a.load(Ordering::Relaxed, &guard); // relaxed load too weak!
//!     atomic::fence(Ordering::Acquire);          // we can now deref the GuardedArc
//!     b = *c;                                    // the refcount is untouched
//! }                                           
//! assert_eq!(b, 1234);
//! 
//! ```
//! 
//! A `GuardedArc` may turned into a full `Arc` that can outlive the `Guard`
//! at the cost of a reference count increment.
//! 
//! ```
//! use std::sync::atomic::Ordering;
//! use crossbeam::epoch;
//! 
//! let a = carc::AtomicArc::new(1234);
//! let mut arc : std::sync::Arc<usize>;
//! {
//!     let guard = epoch::pin();
//!     let guarded_arc = a.load(Ordering::Acquire, &guard);
//!     arc = unsafe { guarded_arc.as_arc() }; // increment the refcount to outlive the guard
//! }
//! assert_eq!(*arc, 1234);
//! ```
//! 
//! When the `AtomicArc` is written to, the old value is placed in a 
//! `DeferredArc`.  When `drop`ped it registers a reference count decrement
//! for deferred execution by the garbage collector.
//! 
//! ```
//! use std::sync::atomic::Ordering;
//! use crossbeam::epoch;
//! 
//! let a = carc::AtomicArc::new(1234);
//! let deferred_arc = a.swap(std::sync::Arc::new(5678), Ordering::AcqRel); 
//! assert_eq!(*deferred_arc, 1234);
//! 
//! ```
//! 
//! A `DeferredArc` can be consumed by an `AtomicArc` write, letting us move
//! things around without touching the reference count
//! 
//! ```
//! use std::sync::atomic::Ordering;
//! use crossbeam::epoch;
//! 
//! let a = carc::AtomicOptionArc::new(1234);
//! let b = carc::AtomicOptionArc::default();
//! let c;
//! {
//!     let option_deferred_arc = a.swap(carc::DeferredArc::none(), Ordering::Acquire);
//!     b.store(option_deferred_arc, Ordering::Release);
//!     let guard = epoch::pin();
//!     c = *b.load(Ordering::Acquire, &guard).unwrap();
//! }
//! assert_eq!(c, 1234);
//! ```
//! 
//! # Deferred reference counting
//! 
//! The key problem is to load a pointer, dereference it to obtain the control
//! block (`ArcInner<T>`), and increment the reference count, without another
//! thread destroying the control block between the read and the increment.
//! 
//! This crate's approach is to use `crossbeam::epoch` to defer decrements of
//! pointers loaded from atomics until the current epoch has been vacated by
//! all threads.  We do this by providing two types, `DeferredArc` and
//! `GuardedArc`, which fulfill for `Arc`s similar roles to those 
//! epoch::Owned` and epoch::Shared` play for `Box`es.
//! 
//! `GuardedArc` results from loads or failed compare_and_sets that read the
//! `AtomicArc` but do not modify it.  They do no share ownership of the
//! control block, but can safely use it for the lifetime of the
//! `epoch::Guard`.  They can be upgraded to a regular `Arc` by incrementing
//! the reference count.  They can be freely copied and trivially dropped.
//! 
//! `DeferredArc` results from swaps and successful compare_and_sets that
//! write to the `AtomicArc`.  The `DeferredArc` is now responsible for the +1 
//! reference count previously owned by the `AtomicArc`.  Other threads may
//! have `GuardedArc`s pointing to the same control block, so the reference
//! count must not be decremented until all threads have left the current
//! epoch.  This is exactly the same requirement as epoch garbage collection,
//! so we implement `Drop` to use `crossbeam::epoch::Guard::defer` to defer
//! the reference count until it is safe to do so.  Though the `GuardedArc`
//! shares ownership of the control block, we cannot directly convert it to an
//! `Arc`, as that `Arc` might be destroyed prematurely.  We can produce an
//! independent by incrementing the reference count.  We can also put a
//! `DeferredArc` back into an `AtomicArc`, removing the need to derefernce it
//! at all.  `DeferredArc`s can be freely moved around outside `pin`nned
//! `epoch`s; they can be cloned only at the same cost as producing an `Arc`
//! which is usually more desireable.  `Drop` will call `pin` which is cheap
//! if the thread is already `pin`ned; if a `Guard` is available we can call
//! `into_guarded` to consume the `DeferredArc`.
//! 
//! The lifetime restrictions on `GuardedArc` constitute a simple form of
//! _coalesced reference counting_.  A `GuardedArc` would otherwise emit an
//! increment and a decrement, but its lifetime restrictions let us cancel
//! these operations out.

// Todo:
//   * Tags
//   * Weak
//   * DeferredArc design space
//       * Hold a &guard (old design)
//       * Drop calls pin (new design, costly?)
//       * Panic on drop, force user to provide guard and convert to something else
//   * Safety design space
//       * SafeAtomicArc vs AtomicArc vs UnsafeAtomicArc which two?
//   * Standard traits for AtomicArc and AtomicOptionArc
//       * Is there any way to get Atomic<Arc<T>> and Atomic<Option<Arc<T>>
//         as 'specializations'?
//   * casting NonZero and NonNull back and forth
//       * Are they worth the trouble?
//       * AtomicNonNullConst?  Why no AtomicPtrConst (because pointers already
//         unsafe?)
//   * Creation trait covariance
//       * Standardize traits on integers or pointers, not both
//   * Safe wrappers or just safe examples?
//       * ArcSwap (only swap and store arcs, AcqRel, never deferred, significant gain)
//       * ArcCell (only set and get arcs, AcqRel)
//       * SafeAtomicArc (does AcqRel memory orderings)
//   * Safety: who takes responsibility?  We need to manipulate reference
//     counts and deref often and far from responsible calls.  Make the atomic
//     operations themselves unsafe since it is the memory ordering they
//     specify that is the root problem.

extern crate crossbeam;

use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::Deref;
use std::option::Option;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use crossbeam::epoch::{Guard, CompareAndSetOrdering};
use crossbeam::atomic::AtomicConsume;


/// Directly manipulate the reference count of an `Arc<T>`
/// 
/// # Safety
/// 
/// The pointer must have originated from `Arc::into_raw`.  The reference count
/// must be at least one.  Memory must have been appropriately synchronized.
/// 
/// # Examples
/// 
/// ```
/// use std::sync::Arc;
/// 
/// unsafe {
///     let p = Arc::into_raw(Arc::new(1234));
///     carc::incr_strong_count(p);
///     let a = Arc::from_raw(p); // This is getting out of hand!  Now there
///     let b = Arc::from_raw(p); // are two of them!
/// }
/// ```
pub unsafe fn incr_strong_count<T>(ptr: *const T) -> *const T {
    let x = Arc::from_raw(ptr); 
    std::mem::forget(x.clone()); // <-- increments strong count
    std::mem::forget(x);
    ptr
}

/// Directly manipulate the reference count of an `Arc<T>`
///
/// # Safety
/// 
/// The pointer must have originated from `Arc::into_raw`.  The reference count
/// must be at least one.  Memory must have been appropriately synchronized.
/// 
/// # Examples
/// 
/// ```
/// use std::sync::Arc;
/// 
/// unsafe {
///     let p = Arc::into_raw(Arc::new(1234));
///     carc::decr_strong_count(p); // Destroys the arc; *p now invalid
/// }
/// ```
pub unsafe fn decr_strong_count<T>(ptr: *const T) -> *const T{
    std::mem::drop(Arc::from_raw(ptr)); // <-- decrements strong count
    ptr
}


/// Model a non-null `*const T`
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct NonNullConst<T: ?Sized> {
    data: std::ptr::NonNull<T>,
}

impl<T: ?Sized> NonNullConst<T> {

    /// Creates a new `NonNullConst`.
    ///
    /// # Safety
    /// 
    /// `ptr` must be non-null.
    /// 
    /// # Examples
    /// 
    pub unsafe fn new_unchecked(ptr: *const T) -> Self {
        Self { data: std::ptr::NonNull::new_unchecked(ptr as *mut T), }
    }

    /// Creates a `NonNullConst` if `ptr` is non-null.
    pub fn new(ptr: *const T) -> Option<Self> {
        std::ptr::NonNull::new(ptr as *mut T).map(|x| { Self { data: x } } )
    }


    /// Returns the underlying non-null pointer
    pub fn as_ptr(self) -> *const T {
        std::ptr::NonNull::as_ptr(self.data)
    }

    /// Dereferences the content
    pub unsafe fn as_ref(&self) -> &T {
        std::ptr::NonNull::as_ref(&self.data)
    }

}

impl<T> From<&T> for NonNullConst<T> {
    fn from(x: &T) -> NonNullConst<T> {
        unsafe { NonNullConst::new_unchecked(x as *const T) }
    }
}

impl<T> From<&mut T> for NonNullConst<T> {
    fn from(x: &mut T) -> NonNullConst<T> {
        unsafe { NonNullConst::new_unchecked(x as *const T) }
    }
}


/// Type that shares ownership of ArcInner
pub trait Owning {}

/// Type that does not share ownership of ArcInner
pub trait NonOwning {}


// OptionArc + Arc -> usize
// Arc -> NonZeroUsize
// usize -> Option
// NonZeroUsize -> OptionArc + Arc
//
// OptionArcIn
// OptionArcOut

/// Can be converted into a pointer (or usize) ultimately from from `Arc::into_raw` or `Arc::as_ptr` or `ptr::null`
pub trait OptionArcLike<T> {

    fn as_usize(&self) -> usize;
    fn into_usize(self) -> usize;

    fn as_ptr(&self) -> *const T { 
        self.as_usize() as *const T
    }
    
}

/// Can be converted into a non-null pointer (or non-zero usize) ultimately from `Arc::into_raw` or `Arc::as_ptr`
pub trait ArcLike<T> : OptionArcLike<T> {

    fn as_non_zero_usize(&self) -> NonZeroUsize;
    fn into_non_zero_usize(self) -> NonZeroUsize;

    fn as_non_null_const(&self) -> NonNullConst<T> {
        // Safety: pointer comes from a nonzero usize
        unsafe { NonNullConst::new_unchecked(self.as_non_zero_usize().get() as *const T) }
    }

}

/// Can be constructed from a non-null pointer (or non-zero usize) ultimately from `Arc::into_raw` or `Arc::as_ptr`
pub trait FromNonZeroUsize<'g, T> {
    unsafe fn from_non_zero_usize(data: NonZeroUsize) -> Self;
}

/// Can be constructed from a pointer (or usize) ultimately from `Arc::into_raw` or `Arc::as_ptr` or `ptr::null`
pub trait FromUsize<'g, T> : FromNonZeroUsize<'g, T> {
    unsafe fn from_usize(data: usize) -> Self;   
}


// Disable safety for raw pointers
impl<T> Owning for *const T {}
impl<T> NonOwning for *const T {}

impl<T> OptionArcLike<T> for *const T {   

    fn as_usize(&self) -> usize {
        *self as usize
    }

    fn into_usize(self) -> usize {
        self as usize
    }

}

impl<'g, T> FromNonZeroUsize<'g, T> for *const T {

    unsafe fn from_non_zero_usize(data: NonZeroUsize) -> Self {
        data.get() as *const T
    }

}

impl<'g, T> FromUsize<'g, T> for *const T {

    unsafe fn from_usize(data: usize) -> Self {
        data as *const T
    }

}


impl<T> Owning for Arc<T> {}

impl<T> ArcLike<T> for Arc<T> {
    
    fn as_non_zero_usize(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(Arc::as_ptr(self) as usize) }
    }

    fn into_non_zero_usize(self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(Arc::into_raw(self) as usize) }
    }

}

impl<T> OptionArcLike<T> for Arc<T> {

    fn as_usize(&self) -> usize {
        Arc::as_ptr(self) as usize
    }

    fn into_usize(self) -> usize {
        Arc::into_raw(self) as usize
    }

}

impl<'g, T> FromNonZeroUsize<'g, T> for Arc<T> {
    
    unsafe fn from_non_zero_usize(data: NonZeroUsize) -> Self {
        Arc::from_raw(data.get() as *const T)
    }

}


impl<T> Owning for Option<Arc<T>> {}

impl<T> OptionArcLike<T> for Option<Arc<T>> {

    fn as_usize(&self) -> usize {
        self.as_ref().map_or(0, |x| { Arc::as_ptr(x) as usize }) 
    }

    fn into_usize(self) -> usize {
        self.map_or(0, |x| { Arc::into_raw(x) as usize })
    }

}

impl<'g, T> FromNonZeroUsize<'g, T> for Option<Arc<T>> {

    unsafe fn from_non_zero_usize(data: NonZeroUsize) -> Self {
        Some(Arc::from_raw(data.get() as *const T))
    }

}

impl<'g, T> FromUsize<'g, T> for Option<Arc<T>> {

    unsafe fn from_usize(data: usize) -> Self {
        match data {
            0 => None,
            data => Some(Arc::from_raw(data as *const T))
        }
    }

}








/// A pointer without shared ownership of T, valid for the current epoch
#[derive(Debug)]
pub struct GuardedArc<'g, T: 'g> {
    data: NonZeroUsize,
    _marker: PhantomData<(&'g(), *const T)>,
}

impl<'g, T: 'g> GuardedArc<'g, T> {

    pub fn none() -> Option<GuardedArc<'g, T>> {
        None
    }

    pub fn try_unwrap(self) -> Result<T, GuardedArc<'g, T>> {
        Err(self)
    }

    pub fn into_raw(self) -> *const T {
        self.data.get() as *const T
    }

    pub fn as_ptr(&self) -> *const T {
        self.data.get() as *const T
    }

    pub unsafe fn from_raw(ptr: *const T) -> Self {
        assert!(!ptr.is_null());
        Self {
            data: NonZeroUsize::new_unchecked(ptr as usize),
            _marker: PhantomData,
        }
    }

    pub fn downgrade(&self) -> std::sync::Weak<T> {
        let x = unsafe { Arc::from_raw(self.as_ptr()) };
        let y = Arc::downgrade(&x);
        std::mem::forget(x);
        y
    }

    pub fn weak_count(&self) -> usize {
        let x = unsafe { Arc::from_raw(self.as_ptr()) };
        let y = Arc::weak_count(&x);
        std::mem::forget(x);
        y
    }

    pub fn strong_count(&self) -> usize {
        let x = unsafe { Arc::from_raw(self.as_ptr()) };
        let y = Arc::strong_count(&x);
        std::mem::forget(x);
        y
    }

    pub fn ptr_eq<'h>(&self, other: GuardedArc<'h, T>) -> bool {
        self.data == other.data
    }

    pub fn get_mut(&mut self) -> Option<&mut T> {
        None
    }



    pub unsafe fn as_arc(&self) -> Arc<T> {
        let a = Arc::from_raw(self.data.get() as *const T);
        std::mem::forget(a.clone());
        a
    }

    pub unsafe fn into_arc(self) -> Arc<T> {
        self.as_arc()
    }

}

impl<'g, T: 'g + Clone> GuardedArc<'g, T> {
    pub fn make_mut(&self) -> UniqueArc<T> {
        UniqueArc::new(self.deref().clone())
    }
}

impl<'g, T> AsRef<T> for GuardedArc<'g, T> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<'g, T> std::borrow::Borrow<T> for GuardedArc<'g, T> {
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<'g, T> Clone for GuardedArc<'g, T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            _marker: PhantomData,
        }
    }
}

impl<'g, T> Copy for GuardedArc<'g, T> {}

impl<'g, T> Deref for GuardedArc<'g, T> {
    
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*(self.data.get() as *const T) }
    }

}

impl<'g, T: Eq> Eq for GuardedArc<'g, T> {}

impl<'g, 'h, T: PartialEq> PartialEq<GuardedArc<'h, T>> for GuardedArc<'g, T> {
    fn eq(&self, other: &GuardedArc<'h, T>) -> bool {
        **self == **other
    }
}



impl<'g, T> NonOwning for GuardedArc<'g, T> {}

impl<'g, T> ArcLike<T> for GuardedArc<'g, T> {
    
    fn as_non_zero_usize(&self) -> NonZeroUsize {
        self.data
    }

    fn into_non_zero_usize(self) -> NonZeroUsize {
        self.as_non_zero_usize()
    }

}

impl<'g, T> OptionArcLike<T> for GuardedArc<'g, T> {

    fn as_usize(&self) -> usize {
        self.data.get()
    }

    fn into_usize(self) -> usize {
        self.data.get()
    }

}

impl<'g, T> FromNonZeroUsize<'g, T> for GuardedArc<'g, T> {

    unsafe fn from_non_zero_usize(data: NonZeroUsize) -> Self {
        Self {
            data,
            _marker: PhantomData
        }
    }
    
}

impl<'g, T> NonOwning for Option<GuardedArc<'g, T>> {}

impl<'g, T> OptionArcLike<T> for Option<GuardedArc<'g, T>> {

    fn as_usize(&self) -> usize {
        self.map_or(0, |x| { x.data.get() as usize })
    }

    fn into_usize(self) -> usize {
        self.as_usize()
    }

}

impl<'g, T> FromUsize<'g, T> for Option<GuardedArc<'g, T>> {

    unsafe fn from_usize(data: usize) -> Self {
        match data {
            0 => None,
            data => Some(GuardedArc::from_non_zero_usize(NonZeroUsize::new_unchecked(data)))
        }
    }

}

impl<'g, T> FromNonZeroUsize<'g, T> for Option<GuardedArc<'g, T>> {

    unsafe fn from_non_zero_usize(data: NonZeroUsize) -> Self {
        Some(GuardedArc::from_non_zero_usize(data))
    }

}





#[derive(Debug)]
pub struct DeferredArc<T> {
    data: NonZeroUsize,
    _marker: PhantomData<T>,
}

impl<T> DeferredArc<T> {

    pub fn none() -> Option<DeferredArc<T>> {
        None
    }

    pub fn new(value: T) -> Self {
        Self {
            data: unsafe { NonZeroUsize::new_unchecked(Arc::into_raw(Arc::new(value)) as usize) },
            _marker: PhantomData,
        }
    }

    pub fn try_unwrap(self) -> Result<T, Self> {
        Err(self)
    }

    pub fn into_raw(self) -> *const T {
        let p = self.as_ptr();
        std::mem::forget(self);
        p
    }

    pub fn as_ptr(&self) -> *const T{
        self.data.get() as *const T        
    }

    pub unsafe fn from_raw(ptr: *const T) -> Self {
        assert!(!ptr.is_null());
        Self {
            data: NonZeroUsize::new_unchecked(ptr as usize),
            _marker: PhantomData,
        }
    }

    pub fn downgrade(&self) -> std::sync::Weak<T> {
        let x = unsafe { Arc::from_raw(self.data.get() as *const T) };
        let y = Arc::downgrade(&x);
        std::mem::forget(x);
        y
    }

    pub fn weak_count(&self) -> usize {
        let x = unsafe { Arc::from_raw(self.data.get() as *const T) };
        let y = Arc::weak_count(&x);
        std::mem::forget(x);
        y
    }

    pub fn strong_count(&self) -> usize {
        let x = unsafe { Arc::from_raw(self.data.get() as *const T) };
        let y = Arc::strong_count(&x);
        std::mem::forget(x);
        y
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.as_ptr() == other.as_ptr()
    }

    pub fn get_mut(&self) -> Option<&mut T> {
        None
    }








    pub unsafe fn as_arc(&self) -> Arc<T> {
        let ptr = self.data.get() as *const T;
        incr_strong_count(ptr);
        Arc::from_raw(ptr)
    }

    pub fn as_guarded<'g>(&self, _guard: &'g Guard) -> GuardedArc<'g, T> {
        GuardedArc {
            data: self.data,
            _marker: PhantomData,
        }
    }

    
    pub unsafe fn into_arc(self) -> Arc<T> {
        self.as_arc() // increments
        // implicit drop decrements after epoch
    }

    pub fn into_guarded<'g>(self, guard: &'g Guard) -> GuardedArc<'g, T> {
        let data = self.data;
        unsafe { 
            guard.defer_unchecked(move || {
                decr_strong_count(data.get() as *const T)
            });
        }
        GuardedArc { data, _marker: PhantomData }
    }

    // get_mut can never succeed
    
    // make_mut can perform a deep copy and return a &mut T once, but
    // subsequent calls will have to do the same again rather than return
    // the same object


}

impl<T: Clone> DeferredArc<T> {
    
    pub fn make_mut(&self) -> UniqueArc<T> {
        UniqueArc::new(self.deref().clone())
    }

}

impl<T> AsRef<T> for DeferredArc<T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.as_ptr() }
    }
}

impl<T> std::borrow::Borrow<T> for DeferredArc<T> {
    fn borrow(&self) -> &T {
        unsafe { &*self.as_ptr() }
    }
}

impl<T> Clone for DeferredArc<T> {
    fn clone(&self) -> Self {
        unsafe { incr_strong_count(self.data.get() as *const T) };
        Self {
            data: self.data,
            _marker: PhantomData,
        }
    }
}

impl<T: Default> Default for DeferredArc<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T> Deref for DeferredArc<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}


/// Performs a pin, which is cheap if we are already pinned and necessary if we aren't
/// If you already have a guard, call into_guarded instead
impl<T> Drop for DeferredArc<T> {
    fn drop(&mut self) {
        let data = self.data;
        unsafe { 
            crossbeam::epoch::pin().defer_unchecked(
                move || { 
                    decr_strong_count(data.get() as *const T)
                } 
            ) 
        }
    }
}
impl<T> std::cmp::Eq for DeferredArc<T> {}

impl<T> From<Arc<T>> for DeferredArc<T> {
    fn from(x: Arc<T>) -> DeferredArc<T> {
        unsafe { DeferredArc::from_non_zero_usize(x.into_non_zero_usize()) }
    }
}

impl<T> From<T> for DeferredArc<T> {
    fn from(x: T) -> DeferredArc<T> {
        DeferredArc::new(x)
    }
}

impl<T> std::cmp::PartialEq<DeferredArc<T>> for DeferredArc<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
} 

unsafe impl<T: Send + Sync> Send for DeferredArc<T> {}
unsafe impl<T: Send + Sync> Sync for DeferredArc<T> {}


impl<T> Owning for DeferredArc<T> {}

impl<'g, T> ArcLike<T> for DeferredArc<T> {

    fn as_non_zero_usize(&self) -> NonZeroUsize {
        self.data
    }

    fn into_non_zero_usize(self) -> NonZeroUsize {
        let DeferredArc { data, .. } = self;
        data
    }

}

impl<'g, T> OptionArcLike<T> for DeferredArc<T> {
    
    fn as_usize(&self) -> usize {
        self.data.get()
    }

    fn into_usize(self) -> usize {
        self.into_non_zero_usize().get()
    }
    
}

impl<'g, T> FromNonZeroUsize<'g, T> for DeferredArc<T> {

    unsafe fn from_non_zero_usize(data: NonZeroUsize) -> Self {
        Self {
            data,
            _marker: PhantomData
        }
    }

}

impl<'g, T> Owning for Option<DeferredArc<T>> {}

impl<'g, T> OptionArcLike<T> for Option<DeferredArc<T>> {

    fn as_usize(&self) -> usize {
        self.as_ref().map_or(0, |x| { x.data.get() })
    }

    fn into_usize(self) -> usize {
        self.map_or(0, |x| { let DeferredArc { data, .. } = x; data.get() })
    }

}

impl<'g, T> FromNonZeroUsize<'g, T> for Option<DeferredArc<T>> {

    unsafe fn from_non_zero_usize(data: NonZeroUsize) -> Self {
        Some(DeferredArc::from_non_zero_usize(data))
    }

}

impl<'g, T> FromUsize<'g, T> for Option<DeferredArc<T>> {
    
    unsafe fn from_usize(data: usize) -> Self {
        match data {
            0 => None,
            data => Some(DeferredArc::from_non_zero_usize(NonZeroUsize::new_unchecked(data)))
        }
    }
    
}




pub struct AtomicNonZeroUsize {
    data: AtomicUsize,
}

impl AtomicNonZeroUsize {
    
    pub fn new(value: NonZeroUsize) -> Self {
        Self { data: AtomicUsize::new(value.get()) }
    }

    pub fn into_inner(self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.data.into_inner()) }
    }

    pub fn get_mut(&mut self) -> &mut NonZeroUsize {
        unsafe { &mut *(self.data.get_mut() as *mut usize as *mut NonZeroUsize) }
    }

    pub fn load(&self, order: Ordering) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.data.load(order)) }
    }

    pub fn load_consume(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.data.load_consume()) }
    }

    pub fn swap(&self, value: NonZeroUsize, order: Ordering) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.data.swap(value.get(), order)) }
    }

    pub fn compare_exchange_weak(
        &self, 
        current: NonZeroUsize,
        new: NonZeroUsize,
        success: Ordering,
        failure: Ordering
    ) -> Result<NonZeroUsize, NonZeroUsize> {
        match self.data.compare_exchange_weak(
            current.get(),                    
            new.get(),
            success,
            failure
        ) {
            Ok(old) => Ok(unsafe { NonZeroUsize::new_unchecked(old) } ),
            Err(current) => Err(unsafe { NonZeroUsize::new_unchecked(current) } ),
        }
    }

    pub fn compare_exchange(
        &self, 
        current: NonZeroUsize,
        new: NonZeroUsize,
        success: Ordering,
        failure: Ordering
    ) -> Result<NonZeroUsize, NonZeroUsize> {
        match self.data.compare_exchange(
            current.get(),                    
            new.get(),
            success,
            failure
        ) {
            Ok(old) => Ok(unsafe { NonZeroUsize::new_unchecked(old) } ),
            Err(current) => Err(unsafe { NonZeroUsize::new_unchecked(current) } ),
        }
    }

}


pub struct CompareAndSetError<T, U> {
    pub current: T,
    pub new: U,
}


pub struct AtomicArc<T> {
    data: AtomicNonZeroUsize,
    _marker: PhantomData<Arc<T>>,
}

impl<T> AtomicArc<T> {

    pub fn new(value: T) -> Self {
        Self {
            data: AtomicNonZeroUsize::new(Arc::new(value).into_non_zero_usize()),
            _marker: PhantomData,
        }
    }

    pub fn into_inner(self) -> Arc<T> {
        unsafe { Arc::from_raw(self.data.into_inner().get() as *const T) }
    }

    pub fn load<'g>(&self, order: Ordering, _guard: &'g Guard) -> GuardedArc<'g, T> {
        unsafe { GuardedArc::from_non_zero_usize(self.data.load(order)) }
    }

    pub fn load_consume<'g>(&self, _guard: &'g Guard) -> GuardedArc<'g, T> {
        unsafe { GuardedArc::from_non_zero_usize(self.data.load_consume()) }
    }

    pub fn store<'g, N: ArcLike<T> + Owning>(&self, new: N, order: Ordering) {
        let order = match order {
            Ordering::Release => Ordering::AcqRel,
            order => order
        };
        self.swap(new, order);
    }

    pub fn swap<'g, N: ArcLike<T> + Owning>(&self, new: N, order: Ordering) -> DeferredArc<T> {
        unsafe { DeferredArc::from_non_zero_usize(self.data.swap(new.into_non_zero_usize(), order)) }
    }
    
    pub fn compare_and_set<'g, C, N, O>(
        &self, 
        current: C, 
        new: N, 
        order: O, 
        _guard: &'g Guard
    ) -> Result<DeferredArc<T>, CompareAndSetError<GuardedArc<'g, T>, N>> where
        C: ArcLike<T> + NonOwning,
        N: ArcLike<T> + Owning,
        O: CompareAndSetOrdering,
    {
        // Safety:
        // If the operation fails, the owned new value is reconstructed and returned to the caller
        // If the operation succeeds, the owned old value is returned to the caller
        match self.data.compare_exchange(
            current.as_non_zero_usize(),
            new.as_non_zero_usize(),
            order.success(),
            order.failure()
        ) {
            Ok(old) => Ok(unsafe {
                new.into_non_zero_usize();
                DeferredArc::from_non_zero_usize(old)
            }),
            Err(current) => Err(
                CompareAndSetError {
                    current: unsafe { GuardedArc::from_non_zero_usize(current) },
                    new,
                }
            ),
        }        
    }

    pub fn compare_and_set_weak<'g, C, N, O>(
        &self, 
        current: C, 
        new: N,
        order: O, 
        _guard: &'g Guard
    ) -> Result<DeferredArc<T>, CompareAndSetError<GuardedArc<'g, T>, N>> where 
        C: ArcLike<T> + NonOwning,
        N: ArcLike<T> + Owning,
        O: CompareAndSetOrdering,
    {
        // Safety:
        // If the operation fails, the owned new value is reconstructed and returned to the caller
        // If the operation succeeds, the owned old value is returned to the caller        
        match self.data.compare_exchange_weak(
            current.as_non_zero_usize(),
            new.as_non_zero_usize(),
            order.success(),
            order.failure()
        ) {
            Ok(old) => Ok(unsafe { 
                new.into_non_zero_usize();
                DeferredArc::from_non_zero_usize(old) 
            }),
            Err(current) => Err(CompareAndSetError {
                current: unsafe { GuardedArc::from_non_zero_usize(current) },
                new,
            }),
        }        
    }

}














pub struct AtomicOptionArc<T> {
    data: AtomicUsize,
    _marker: PhantomData<Option<Arc<T>>>,
}

impl<T> AtomicOptionArc<T> {

    pub const fn none() -> Self {
        Self {
            data: AtomicUsize::new(0),
            _marker: PhantomData,
        }
    }

    pub fn new(value: T) -> Self {
        Self {
            data: AtomicUsize::new(Arc::new(value).into_non_zero_usize().get()),
            _marker: PhantomData,
        }
    }

    pub fn load<'g>(&self, order: Ordering, _guard: &'g Guard) -> Option<GuardedArc<'g, T>> {
        unsafe { Option::from_usize(self.data.load(order)) }
    }

    pub fn load_consume<'g>(&self, _guard: &'g Guard) -> Option<GuardedArc<'g, T>> {
        unsafe { Option::from_usize(self.data.load_consume()) }
    }

    pub fn store<'g, N: OptionArcLike<T> + Owning>(&self, new: N, order: Ordering) {
        let order = match order {
            Ordering::Release => Ordering::AcqRel,
            order => order
        };
        self.swap(new, order);
    }

    pub fn swap<'g, N: OptionArcLike<T> + Owning>(&self, new: N, order: Ordering) -> Option<DeferredArc<T>> {
        unsafe { Option::from_usize(self.data.swap(new.into_usize(), order)) }
    }

    pub fn compare_and_set<'g, C: OptionArcLike<T> + NonOwning, N: OptionArcLike<T> + Owning, O: CompareAndSetOrdering>(
        &self, 
        current: C, 
        new: N, 
        order: O,
        _guard: &'g Guard
    ) -> Result<Option<DeferredArc<T>>, CompareAndSetError<Option<GuardedArc<'g, T>>, N>> {
        // Safety:
        // If the operation fails, the owned new value is reconstructed and returned to the caller
        // If the operation succeeds, the owned old value is returned to the caller
        match self.data.compare_exchange(
            current.as_usize(),
            new.as_usize(),
            order.success(),
            order.failure()
        ) {
            Ok(old) => unsafe {
                new.into_usize();
                Ok(Option::from_usize(old))
            },
            Err(current) => Err(CompareAndSetError {
                current: unsafe { Option::from_usize(current) },
                new,
            }),
        }        
    }

    pub fn compare_and_set_weak<'g, C: OptionArcLike<T> + NonOwning, N: OptionArcLike<T> + Owning, O: CompareAndSetOrdering>(
        &self, 
        current: C, 
        new: N, 
        order: O,
        _guard: &'g Guard
    ) -> Result<Option<DeferredArc<T>>, CompareAndSetError<Option<GuardedArc<'g, T>>, N>> {
        // Safety:
        // If the operation fails, the owned new value is reconstructed and returned to the caller
        // If the operation succeeds, the owned old value is returned to the caller
        match self.data.compare_exchange_weak(
            current.as_usize(),
            new.as_usize(),
            order.success(),
            order.failure()
        ) {
            Ok(old) => unsafe {
                new.into_usize();
                Ok(Option::from_usize(old))
            },
            Err(current) => Err(CompareAndSetError {
                current: unsafe { Option::from_usize(current) },
                new,
            }),
        }        
    }

    pub fn into_inner(self) -> Option<Arc<T>> {
        let x = self.data.into_inner();        
        match x {
            0 => None,
            x => Some(unsafe { Arc::from_raw(x as *const T) } ),
        }
    }

}

impl<T> Default for AtomicOptionArc<T> {
    fn default() -> Self {
        Self {
            data: AtomicUsize::new(0),
            _marker: PhantomData,
        }
    }
}




/// A `Box`-like object that is ready to turn into an `Arc` when required
/// 
/// This is a useful initial state for concurrent data structure nodes which
/// need to be mutated until they are published
#[derive(Debug)]
pub struct UniqueArc<T> {
    data: Arc<T>,
}

impl<T> UniqueArc<T> {

    pub fn new(value: T) -> Self {
        Self {
            data: Arc::new(value),
        }
    }

    pub fn unwrap(self) -> T {
        match Arc::try_unwrap(self.data) {
            Ok(value) => value,
            Err(_) => panic!()
        }
    }

    pub fn into_raw(self) -> *mut T {
        Arc::into_raw(self.data) as *mut T
    }

    /// # Safety
    /// 
    /// Must be an Arc pointer AND must be unique (checkable with get_mut
    /// debug mode) AND must not be deferred
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        let mut data = Arc::from_raw(ptr);
        assert!(Arc::get_mut(&mut data).is_some());
        Self { data }
    }

}

impl<T> AsMut<T> for UniqueArc<T> {
    fn as_mut(&mut self) -> &mut T {
        // Arc::get_mut_unchecked is nightly-only
        unsafe { &mut *(Arc::as_ptr(&self.data) as *mut T) }
    }
}

impl<T> AsRef<T> for UniqueArc<T> {
    fn as_ref(&self) -> &T {
        self.data.as_ref()
    }
}

impl<T> std::borrow::Borrow<T> for UniqueArc<T> {
    fn borrow(&self) -> &T {
        self.data.as_ref()
    }
}

impl<T> std::borrow::BorrowMut<T> for UniqueArc<T> {
    fn borrow_mut(&mut self) -> &mut T {
        self.as_mut()
    }
}

impl<T: Clone> Clone for UniqueArc<T> {
    fn clone(&self) -> Self {
        Self::new(self.as_ref().clone())
    }
}

impl<T: Default> Default for UniqueArc<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T> Deref for UniqueArc<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.data.deref()
    }
}

impl<T> std::ops::DerefMut for UniqueArc<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.as_mut()
    }
}

impl<T> From<T> for UniqueArc<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T> Owning for UniqueArc<T> {}

impl<T> OptionArcLike<T> for UniqueArc<T> {    

    fn as_usize(&self) -> usize {
        self.as_ptr() as usize
    }
    
    fn into_usize(self) -> usize {
        Arc::into_raw(self.data) as usize
    }

}

impl<T> ArcLike<T> for UniqueArc<T> {

    fn as_non_zero_usize(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.as_usize()) }
    }

    fn into_non_zero_usize(self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.into_usize()) }
    }

}

impl<T> Owning for Option<UniqueArc<T>> {}

impl<T> OptionArcLike<T> for Option<UniqueArc<T>> {    

    fn as_usize(&self) -> usize {
        self.as_ref().map_or(0, |x| { x.as_usize() })
    }
    
    fn into_usize(self) -> usize {
        self.map_or(0, |x| { x.into_usize() })
    }

}
















#[cfg(test)]
mod tests {

    //extern crate crossbeam;

    use super::AtomicOptionArc;
    use std::sync::Arc;
    use crossbeam::epoch;
    use std::sync::atomic::Ordering;

    struct IntrusiveStackNode<T> {
        next: *const IntrusiveStackNode<T>,
        value: T,
    }

    unsafe impl<T: Send> Send for IntrusiveStackNode<T> {}
    unsafe impl<T: Sync> Sync for IntrusiveStackNode<T> {}
    
    impl<T: Default> Default for IntrusiveStackNode<T> {
        fn default() -> Self {
            Self::new(T::default())
        }
    }

    impl<T> IntrusiveStackNode<T> {
        fn new(value: T) -> Self {
            Self {
                next: std::ptr::null(),
                value,
            }
        }
    }


    struct IntrusiveStack<T> {
        head: AtomicOptionArc<IntrusiveStackNode<T>>,
    }
    
    impl<T: Send + Sync> IntrusiveStack<T> {

        fn new() -> Self {
            Self {
                head: AtomicOptionArc::none(),
            }
        }

        fn top<'g>(&self, guard: &'g epoch::Guard) -> Option<&'g T> {
            let old = self.head.load(Ordering::Acquire, &guard);
            old.map(|p| { unsafe { &(*p.as_ptr()).value } })
        }
        fn push(&self, new: Arc<IntrusiveStackNode<T>>) {
            let guard = epoch::pin();
            let mut new = new;
            let mut current = self.head.load(Ordering::Relaxed, &guard);
            loop {
                // wart: ownership transferred only if we win, so is the distinction meaningful?
                Arc::get_mut(&mut new).unwrap().next = current.as_ref().map_or(std::ptr::null(), |x| { x.as_ptr() });
                match self.head.compare_and_set_weak(current, new, (Ordering::Release, Ordering::Relaxed), &guard) {
                    Ok(_) => break,
                    Err(b) => { 
                        current = b.current; 
                        new = b.new;
                        continue 
                    },
                }
            }
        }

        fn pop(&self) -> Option<Arc<IntrusiveStackNode<T>>> {
            let guard = epoch::pin();
            let mut current = self.head.load(Ordering::Acquire, &guard);
            loop {
                match current {
                    None => return None,
                    Some(p) => {                        
                        match self.head.compare_and_set_weak(current, p.next, (Ordering::Acquire, Ordering::Acquire), &guard) {
                            Ok(_) => return Some(unsafe {p.as_arc()}),
                            Err(e) => {
                                current = e.current;
                            }
                        }
                    }
                }                
            }
        }

    }

    #[test]
    fn it_works() {
        
        let g = epoch::pin();
        let a = IntrusiveStack::new();
        assert!(a.top(&g).is_none());
        a.push(Arc::new(IntrusiveStackNode::new(7)));
        assert_eq!(a.top(&g).unwrap(), &7);
        let x = a.top(&g);
        assert_eq!(a.pop().unwrap().value, 7);
        assert!(a.top(&g).is_none());
        drop(&g);
        println!("{:?}", x);

    }
}
