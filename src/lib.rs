//! Fast AtomicArc using Crossbeam to coalesce and defer reference counting
//! 
//! # Examples
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
//! with the guard for deferred execution when all other threads have
//! moved on from the current epoch and there are no outstanding `GuardedArc`s
//! 
//! ```
//! use std::sync::atomic::Ordering;
//! use crossbeam::epoch;
//! 
//! let a = carc::AtomicArc::new(1234);
//! let mut arc : std::sync::Arc<usize>;
//! {
//!     let guard = epoch::pin();
//!     // just an atomic swap, reference count untouched
//!     let deferred_arc = a.swap(std::sync::Arc::new(5678), Ordering::AcqRel, &guard); 
//!     // eventually decremented by the Crossbeam garbage collector
//! }
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
//!     let guard = epoch::pin();
//!     let option_deferred_arc = a.swap(std::ptr::null(), Ordering::Acquire, &guard);
//!     b.store(option_deferred_arc, Ordering::Release, &guard);
//!     c = *b.load(Ordering::Acquire, &guard).unwrap();
//! }
//! assert_eq!(c, 1234);
//! ```
//! 
//! 
//! 
//! Todo: tags, Weak

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
    unsafe fn from_non_zero_usize(data: NonZeroUsize, guard: &'g Guard) -> Self;
}

/// Can be constructed from a pointer (or usize) ultimately from `Arc::into_raw` or `Arc::as_ptr` or `ptr::null`
pub trait FromUsize<'g, T> : FromNonZeroUsize<'g, T> {
    unsafe fn from_usize(data: usize, guard: &'g Guard) -> Self;   
}


impl<T> OptionArcLike<T> for *const T {   

    fn as_usize(&self) -> usize {
        *self as usize
    }

    fn into_usize(self) -> usize {
        self as usize
    }

}

impl<'g, T> FromNonZeroUsize<'g, T> for *const T {

    unsafe fn from_non_zero_usize(data: NonZeroUsize, _guard: &'g Guard) -> Self {
        data.get() as *const T
    }

}

impl<'g, T> FromUsize<'g, T> for *const T {

    unsafe fn from_usize(data: usize, _guard: &'g Guard) -> Self {
        data as *const T
    }

}



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
    
    unsafe fn from_non_zero_usize(data: NonZeroUsize, _guard: &'g Guard) -> Self {
        Arc::from_raw(data.get() as *const T)
    }

}


impl<T> OptionArcLike<T> for Option<Arc<T>> {

    fn as_usize(&self) -> usize {
        self.as_ref().map_or(0, |x| { Arc::as_ptr(x) as usize }) 
    }

    fn into_usize(self) -> usize {
        self.map_or(0, |x| { Arc::into_raw(x) as usize })
    }

}

impl<'g, T> FromNonZeroUsize<'g, T> for Option<Arc<T>> {

    unsafe fn from_non_zero_usize(data: NonZeroUsize, _guard: &'g Guard) -> Self {
        Some(Arc::from_raw(data.get() as *const T))
    }

}

impl<'g, T> FromUsize<'g, T> for Option<Arc<T>> {

    unsafe fn from_usize(data: usize, _guard: &'g Guard) -> Self {
        match data {
            0 => None,
            data => Some(Arc::from_raw(data as *const T))
        }
    }

}








/// A pointer without shared ownership of T, valid for the current epoch
#[derive(Debug, Eq, PartialEq)]
pub struct GuardedArc<'g, T: 'g> {
    data: NonZeroUsize,
    _marker: PhantomData<(&'g(), *const T)>,
}

impl<'g, T: 'g> GuardedArc<'g, T> {

    pub unsafe fn as_arc(&self) -> Arc<T> {
        let a = Arc::from_raw(self.data.get() as *const T);
        std::mem::forget(a.clone());
        a
    }

    pub fn as_ptr(&self) -> *const T {
        self.data.get() as *const T
    }

    pub unsafe fn into_arc(self) -> Arc<T> {
        self.as_arc()
    }

    pub fn ptr_eq<'h>(&self, other: GuardedArc<'h, T>) -> bool {
        self.data == other.data
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

    unsafe fn from_non_zero_usize(data: NonZeroUsize, _guard: &'g Guard) -> Self {
        Self {
            data,
            _marker: PhantomData
        }
    }
    
}

impl<'g, T> OptionArcLike<T> for Option<GuardedArc<'g, T>> {

    fn as_usize(&self) -> usize {
        self.map_or(0, |x| { x.data.get() as usize })
    }

    fn into_usize(self) -> usize {
        self.as_usize()
    }

}

impl<'g, T> FromUsize<'g, T> for Option<GuardedArc<'g, T>> {

    unsafe fn from_usize(data: usize, guard: &'g Guard) -> Self {
        match data {
            0 => None,
            data => Some(GuardedArc::from_non_zero_usize(NonZeroUsize::new_unchecked(data), guard))
        }
    }

}

impl<'g, T> FromNonZeroUsize<'g, T> for Option<GuardedArc<'g, T>> {

    unsafe fn from_non_zero_usize(data: NonZeroUsize, guard: &'g Guard) -> Self {
        Some(GuardedArc::from_non_zero_usize(data, guard))
    }

}





#[derive(Debug)]
pub struct DeferredArc<'g, T> {
    data: NonZeroUsize,
    guard: &'g Guard,
    _marker: PhantomData<T>,
}

impl<'g, T> DeferredArc<'g, T> {

    pub unsafe fn as_arc(&self) -> Arc<T> {
        let x : Arc<T> = Arc::from_non_zero_usize(self.data, self.guard);
        std::mem::forget(x.clone());
        x
    }

    pub fn as_guarded(&self) -> GuardedArc<'g, T> {
        GuardedArc {
            data: self.data,
            _marker: PhantomData,
        }
    }

    pub fn as_ptr(&self) -> *const T{
        self.data.get() as *const T        
    }

    pub fn new(value: T, guard: &'g Guard) -> Self {
        unsafe { Self::from_non_zero_usize(Arc::new(value).into_non_zero_usize(), guard) }
    }

    pub unsafe fn into_arc(self) -> Arc<T> {
        self.as_arc() // increments
        // implicit drop decrements after epoch
    }

    pub fn into_guarded(self) -> GuardedArc<'g, T> {
        self.as_guarded()
        // implicit drop decrements after epoch
    }

}

impl<'g, T> AsRef<T> for DeferredArc<'g, T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.as_ptr() }
    }
}

impl<'g, T> std::borrow::Borrow<T> for DeferredArc<'g, T> {
    fn borrow(&self) -> &T {
        unsafe { &*self.as_ptr() }
    }
}

impl<'g, T> Clone for DeferredArc<'g, T> {
    fn clone(&self) -> Self {
        unsafe { std::mem::forget(self.as_arc()) };
        Self {
            data: self.data,
            guard: self.guard,
            _marker: PhantomData,
        }
    }
}

impl<'g, T> Drop for DeferredArc<'g, T> {
    fn drop(&mut self) {
        let x = self.data.get();
        unsafe { 
            self.guard.defer_unchecked(
                move || { 
                    Arc::from_raw(x as *const T)
                } 
            ) 
        }
    }
}

impl<'g, T> Deref for DeferredArc<'g, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'g, T> std::cmp::Eq for DeferredArc<'g, T> {}

impl<'g, T> std::cmp::PartialEq<DeferredArc<'g, T>> for DeferredArc<'g, T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
} 

impl<'g, T> ArcLike<T> for DeferredArc<'g, T> {

    fn as_non_zero_usize(&self) -> NonZeroUsize {
        self.data
    }

    fn into_non_zero_usize(self) -> NonZeroUsize {
        let DeferredArc { data, .. } = self;
        data
    }

}

impl<'g, T> OptionArcLike<T> for DeferredArc<'g, T> {
    
    fn as_usize(&self) -> usize {
        self.data.get()
    }

    fn into_usize(self) -> usize {
        self.into_non_zero_usize().get()
    }
    
}

impl<'g, T> FromNonZeroUsize<'g, T> for DeferredArc<'g, T> {

    unsafe fn from_non_zero_usize(data: NonZeroUsize, guard: &'g Guard) -> Self {
        Self {
            data,
            guard,
            _marker: PhantomData
        }
    }

}


impl<'g, T> OptionArcLike<T> for Option<DeferredArc<'g, T>> {

    fn as_usize(&self) -> usize {
        self.as_ref().map_or(0, |x| { x.data.get() })
    }

    fn into_usize(self) -> usize {
        self.map_or(0, |x| { let DeferredArc { data, .. } = x; data.get() })
    }

}

impl<'g, T> FromNonZeroUsize<'g, T> for Option<DeferredArc<'g, T>> {

    unsafe fn from_non_zero_usize(data: NonZeroUsize, guard: &'g Guard) -> Self {
        Some(DeferredArc::from_non_zero_usize(data, guard))
    }

}

impl<'g, T> FromUsize<'g, T> for Option<DeferredArc<'g, T>> {
    
    unsafe fn from_usize(data: usize, guard: &'g Guard) -> Self {
        match data {
            0 => None,
            data => Some(DeferredArc::from_non_zero_usize(NonZeroUsize::new_unchecked(data), guard))
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

    pub fn load<'g>(&self, order: Ordering, guard: &'g Guard) -> GuardedArc<'g, T> {
        unsafe { GuardedArc::from_non_zero_usize(self.data.load(order), guard) }
    }

    pub fn load_consume<'g>(&self, guard: &'g Guard) -> GuardedArc<'g, T> {
        unsafe { GuardedArc::from_non_zero_usize(self.data.load_consume(), guard) }
    }

    pub fn store<'g, N: ArcLike<T>>(&self, new: N, order: Ordering, guard: &'g Guard) {
        let order = match order {
            Ordering::Release => Ordering::AcqRel,
            order => order
        };
        self.swap(new, order, guard);
    }

    pub fn swap<'g, N: ArcLike<T>>(&self, new: N, order: Ordering, guard: &'g Guard) -> DeferredArc<'g, T> {
        unsafe { DeferredArc::from_non_zero_usize(self.data.swap(new.into_non_zero_usize(), order), guard) }
    }
    
    pub fn compare_and_set<'g, C, N, O>(
        &self, 
        current: C, 
        new: N, 
        order: O, 
        guard: &'g Guard
    ) -> Result<DeferredArc<'g, T>, CompareAndSetError<GuardedArc<'g, T>, N>> where
        C: ArcLike<T>,
        N: ArcLike<T>,
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
                DeferredArc::from_non_zero_usize(old, guard)
            }),
            Err(current) => Err(
                CompareAndSetError {
                    current: unsafe { GuardedArc::from_non_zero_usize(current, guard) },
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
        guard: &'g Guard
    ) -> Result<DeferredArc<'g, T>, CompareAndSetError<GuardedArc<'g, T>, N>> where 
        C: ArcLike<T>,
        N: ArcLike<T>,
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
                DeferredArc::from_non_zero_usize(old, guard) 
            }),
            Err(current) => Err(CompareAndSetError {
                current: unsafe { GuardedArc::from_non_zero_usize(current, guard) },
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

    pub const fn null() -> Self {
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

    pub fn load<'g>(&self, order: Ordering, guard: &'g Guard) -> Option<GuardedArc<'g, T>> {
        unsafe { Option::from_usize(self.data.load(order), guard) }
    }

    pub fn load_consume<'g>(&self, guard: &'g Guard) -> Option<GuardedArc<'g, T>> {
        unsafe { Option::from_usize(self.data.load_consume(), guard) }
    }

    pub fn store<'g, N: OptionArcLike<T>>(&self, new: N, order: Ordering, guard: &'g Guard) {
        let order = match order {
            Ordering::Release => Ordering::AcqRel,
            order => order
        };
        self.swap(new, order, guard);
    }

    pub fn swap<'g, N: OptionArcLike<T>>(&self, new: N, order: Ordering, guard: &'g Guard) -> Option<DeferredArc<'g, T>> {
        unsafe { Option::from_usize(self.data.swap(new.into_usize(), order), guard) }
    }

    pub fn compare_and_set<'g, C: OptionArcLike<T>, N: OptionArcLike<T>, O: CompareAndSetOrdering>(
        &self, 
        current: C, 
        new: N, 
        order: O,
        guard: &'g Guard
    ) -> Result<Option<DeferredArc<'g, T>>, CompareAndSetError<Option<GuardedArc<'g, T>>, N>> {
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
                Ok(Option::from_usize(old, guard))
            },
            Err(current) => Err(CompareAndSetError {
                current: unsafe { Option::from_usize(current, guard) },
                new,
            }),
        }        
    }

    pub fn compare_and_set_weak<'g, C: OptionArcLike<T>, N: OptionArcLike<T>, O: CompareAndSetOrdering>(
        &self, 
        current: C, 
        new: N, 
        order: O,
        guard: &'g Guard
    ) -> Result<Option<DeferredArc<'g, T>>, CompareAndSetError<Option<GuardedArc<'g, T>>, N>> {
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
                Ok(Option::from_usize(old, guard))
            },
            Err(current) => Err(CompareAndSetError {
                current: unsafe { Option::from_usize(current, guard) },
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
                head: AtomicOptionArc::null(),
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
