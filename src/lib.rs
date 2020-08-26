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
//! `DeferredArc` back into an `AtomicArc`, removing the need to dereference it
//! at all.  `DeferredArc`s can be freely moved around outside `pin`nned
//! `epoch`s; they can be cloned only at the same cost as producing an `Arc`
//! which is usually more desirable.  `Drop` will call `pin` which is cheap
//! if the thread is already `pin`ned; if a `Guard` is available we can call
//! `into_guarded` to consume the `DeferredArc`.
//! 
//! The lifetime restrictions on `GuardedArc` constitute a simple form of
//! _coalesced reference counting_.  A `GuardedArc` would otherwise emit an
//! increment and a decrement, but its lifetime restrictions let us cancel
//! these operations out.

// Todo:
//   * Tags
//       impl ArcLike<T> for (Option<ArcLike<T>>, tag) !
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
//   * fetch bitwise operations for tags
//     * example: fetch_or(CANCELLED_BIT)
//   * make a stack with O(1) atomic snapshot

extern crate crossbeam;

use std::marker::{PhantomData, Sized};
use std::mem::{align_of, forget, swap};
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::option::Option;
use std::sync::{Arc, Weak};
use std::sync::atomic::{AtomicUsize, Ordering};
use crossbeam::epoch::{CompareAndSetOrdering};
use crossbeam::atomic::AtomicConsume;



unsafe fn incr_strong_count<T>(ptr: *const T) {
    let x = Arc::from_raw(ptr);
    let y = x.clone(); // clone causes strong increment
    forget(y); // don't let drop decrement
    forget(x); // don't let drop decrement
}

unsafe fn decr_strong_count<T>(ptr: *const T) {
    Arc::from_raw(ptr); // implicit drop causes decrement
}


struct Ledger {

    // TODO: Need a fast small map (ptr: usize) -> (n: isize, fn)
    // maybe static sized array? maybe SmallVec?  maybe SmallFlatMap exists somewhere
    vec: std::vec::Vec<(usize, isize, fn(usize, isize))>,

    // it is benign to spill to earlier action if benchmarking shows that's faster than wrangling a big map


}

impl Ledger {

    fn new() -> Self {
        Self {
            vec: std::vec::Vec::default()            
        }
    }

    pub fn defer_incr<T>(&mut self, ptr: *const T) {
        let ptr = ptr as usize;
        let i = 0;
        while i != self.vec.len() {
            if self.vec[i].0 == ptr { // <-- linear find
                if self.vec[i].1 == -1 {
                    self.vec.swap_remove(i); // <-- remove if it goes to zero
                } else {
                    self.vec[i].1 += 1; // <-- note the increment
                }
                return
            }
            i += 1;
        }
        self.vec.push((ptr, 1, |ptr, n| {
            debug_assert!(n > 0);
            let ptr = ptr as *const T; // <-- function pointer recovers type
            while n > 0 {
                incr_strong_count(ptr);
                n -= 1;
            }
        }));
    }

    pub fn defer_decr<T>(&mut self, ptr: *const T) {
        let ptr = ptr as usize;
        let i = 0;
        while i != self.vec.len() {
            if self.vec[i].0 == ptr { // <-- linear find
                if self.vec[i].1 == 1 {
                    self.vec.swap_remove(i); // <-- remove if it would go to zero
                } else {
                    self.vec[i].1 -= 1; // <-- perform decrement
                }
                return
            }
            i += 1;
        }
        self.vec.push((ptr, 1, |ptr, n| {
            debug_assert!(n < 0);
            let ptr = ptr as *const T; // <-- function pointer recovers type
            while n < 0 {
                decr_strong_count(ptr);
                n += 1;
            }            
        }));
    }

    pub fn perform_increments(&mut self) {
        let i = 0;
        while i != self.vec.len() {
            if self.vec[i].1 > 0 {
                let (p, n, f) = self.vec.swap_remove(i);
                f(p, n);
            } else {
                i += 1;
            }
        }
    }

    pub fn perform_decrements(&mut self) {
        let i = 0;
        while !self.vec.is_empty() {
            let (p, n, f) = self.vec.swap_remove(0);
            debug_assert!(n < 0);    
            f(p, n);
        }
    }

}

struct Guard {
    guard: crossbeam::epoch::Guard,
    ledger: Ledger,

}

pub fn pin() -> Guard {
    Guard { 
        guard: crossbeam::epoch::pin(),
        ledger: Ledger::new(),
    }
}

impl Guard {

    pub fn repin(&mut self) {
        self.ledger.perform_increments();
        self.guard.repin();
        // ok to hold on to the decrements
    }

}

impl Drop for Guard {
    fn drop(&mut self) {
        let Guard {guard, ledger} = self;
        unsafe {
            guard.defer_unchecked(move || {
                ledger.perform_decrements()
            });
        }
    }
}






/// Compile-time max
/// 
/// <https://stackoverflow.com/a/53646925>
const fn max(a: usize, b: usize) -> usize {
    [a, b][(a < b) as usize]
}

/// A bit mask, calculated at compile time, for the alignment bits of a pointer
/// returned from `Arc<T>::into_raw`.
/// 
/// We want the alignment not of an arbitrary T, but a T that is the third
/// field of the Arc<T>'s private control block:
/// 
///     struct ArcInner<T> {
///         strong: AtomicUsize,
///         weak: AtomicUsize,
///         data: T,
///     }
///
/// The alignment of this T is at least that of AtomicUsize.
const fn tag_mask<T>() -> usize {
    // To convert a power-of-two alignment to a mask, subtract one.
    max(align_of::<AtomicUsize>(), align_of::<T>()) - 1
}

/// A bit mask, calculated at compile time, for the non-alignment bits of a
/// pointer returned from `Arc<T>::into_raw`.  This pointer may be more
/// strictly aligned than a general *const T.
const fn ptr_mask<T>() -> usize {
    !tag_mask::<T>()
}


/// Types implementing `ArcLike<T>` can be used as arguments to the methods of
/// `AtomicArc<T>`
pub trait ArcLike<T> {

    fn into_usize(this: Self) -> usize;
    fn as_ptr(this: &Self) -> *const T;
    unsafe fn from_usize(data: usize) -> Self;

}

/// Types implementing `ArcLike<T> + Owning` own an `Arc<T>` and every call to 
/// `ArcLike<T>::into_usize` must be paired with a call to
/// `U::from_usize where U: ArcLike<T> + Owning`.
pub unsafe trait Owning {}

/// Types implementing `ArcLike<T> + NotOwning` don't own an `Arc<T>` and calls
/// to `ArcLike<T>::into_usize` and `ArcLike<T>::from_usize` have no side
/// effects.
pub unsafe trait NotOwning {}

/// Types implementing `ArcLike<T> + NotNull` will never return zero from
/// `ArcLike<T>::into_usize` and it is an error to call
/// `ArcLike<T>::from_usize` with an argument of zero.
pub unsafe trait NotNull {}



impl<T> ArcLike<T> for Arc<T> {

    fn into_usize(this: Self) -> usize {
        let data = Arc::into_raw(this) as usize;
        // Check pointer is non-null
        debug_assert_ne!(data & ptr_mask::<T>(), 0);
        // Check pointer has expected alignment
        debug_assert_eq!(data & tag_mask::<T>(), 0);
        data
    }

    fn as_ptr(this: &Self) -> *const T {
        Arc::as_ptr(this)
    }

    unsafe fn from_usize(data: usize) -> Self {
        // Check pointer part is non-zero
        debug_assert_ne!(data & ptr_mask::<T>(), 0);
        // Check tag bits are zero
        debug_assert_eq!(data & tag_mask::<T>(), 0);
        Arc::from_raw(data as *const T)
    }

}

impl<'g, T> From<GuardedArc<'g, T>> for Arc<T> {
    fn from(value: GuardedArc<'g, T>) -> Self {
        let ptr = GuardedArc::as_ptr(&value);
        unsafe {
            GuardedArc::incr_strong_count(ptr);
            Self::from_raw(ptr)
        }
    }
}

impl<'g, T> From<DeferredArc<T>> for Arc<T> {
    fn from(value: DeferredArc<T>) -> Self {
        let ptr = DeferredArc::as_ptr(&value);
        unsafe {
            DeferredArc::incr_strong_count(ptr);
            Self::from_raw(ptr)
        }
    }
}

impl<'g, T> From<UniqueArc<T>> for Arc<T> {
    fn from(value: UniqueArc<T>) -> Self {
        let UniqueArc { data } = value;
        data
    }
}

unsafe impl<T> Owning for Arc<T> {}
unsafe impl<T> NotNull for Arc<T> {}

impl<T, U: ArcLike<T> + NotNull> ArcLike<T> for Option<U> {
    
    fn into_usize(this: Self) -> usize {
        match this {
            None => 0,
            Some(this) => ArcLike::into_usize(this)
        }
    }

    fn as_ptr(this: &Self) -> *const T {
        match &this {
            None => std::ptr::null(),
            Some(this) => ArcLike::as_ptr(this)
        }
    }
    
    unsafe fn from_usize(data: usize) -> Self {
        match data {
            0    => None,
            data => Some(U::from_usize(data)),
        }
    }

}

unsafe impl<T: Owning> Owning for Option<T> {}
unsafe impl<T: NotOwning> NotOwning for Option<T> {}

impl<T, U: ArcLike<T>> ArcLike<T> for (U, usize) {

    fn into_usize(this: Self) -> usize {
        let (arc, tag) = this;
        U::into_usize(arc) | (tag & tag_mask::<T>())
    }

    fn as_ptr(this: &Self) -> *const T {
        U::as_ptr(&this.0)
    }

    unsafe fn from_usize(data: usize) -> Self {
        (U::from_usize(data & ptr_mask::<T>()), data & tag_mask::<T>())
    }

}

unsafe impl<T: Owning> Owning for (T, usize) {}
unsafe impl<T: NotNull> NotNull for (T, usize) {}
unsafe impl<T: NotOwning> NotOwning for (T, usize) {}


/// The pointer must have been returned by `Arc::into_raw` or be null.  Such
/// pointers may be more strictly aligned than a general `*const T`, and we
/// check for this strict alignment in debug mode
impl<T> ArcLike<T> for *const T {
    
    fn into_usize(this: Self) -> usize {
        let data = this as usize;
        debug_assert_eq!(data & tag_mask::<T>(), 0);
        data
    }

    fn as_ptr(this: &Self) -> *const T {
        *this
    }

    unsafe fn from_usize(data: usize) -> Self {
        debug_assert_eq!(data & tag_mask::<T>(), 0);
        data as Self
    }

}

// We implement all marker traits for *const T to disable checking in `AtomicArc` methods
unsafe impl<T> Owning for *const T {}
unsafe impl<T> NotOwning for *const T {}
unsafe impl<T> NotNull for *const T {}


/// Not crazy!  Allows us to work with tagged nulls
/// 
/// ```
/// struct Node {
///     next: Option<Arc<Node>>,
///     action: FnOnce(),
/// }
/// const LOCKED_NO_WAITERS = 0;
/// const NOT_LOCKED = 1;
/// let head = AtomicOptionArc::<Node>::new(NOT_LOCKED);
/// ```
impl<T> ArcLike<T> for usize {

    fn into_usize(this: Self) -> usize {
        debug_assert_eq!(this & ptr_mask::<T>(), 0);
        this
    }

    fn as_ptr(this: &Self) -> *const T {
        (this & ptr_mask::<T>()) as *const T
    }

    unsafe fn from_usize(data: usize) -> Self {
        debug_assert_eq!(data & ptr_mask::<T>(), 0);
        data
    }

}

// Assume the user knows what they are doing
unsafe impl Owning for usize {}
unsafe impl NotOwning for usize {}






unsafe fn with_arc<T, R, F: FnOnce(&Arc<T>) -> R>(ptr: *const T, f: F) -> R {
    let this = Arc::from_raw(ptr);
    let result = f(&this);
    forget(this);
    result
}

pub trait ArcInterface<T> : ArcLike<T> + Sized {

    fn into_raw(this: Self) -> *const T {
        (Self::into_usize(this) & ptr_mask::<T>()) as *const T
    }

    unsafe fn from_raw(ptr: *const T) -> Self {
        Self::from_usize(ptr as usize)
    }

    fn ptr_eq<Other: ArcInterface<T>>(this: &Self, other: &Other) -> bool {
        Self::as_ptr(this) == Other::as_ptr(&other)
    }

    fn downgrade(this: &Self) -> Weak<T> {
        unsafe { with_arc(Self::as_ptr(this), |x| { Arc::downgrade(x) }) }
    }

    fn strong_count(this: &Self) -> usize {
        unsafe { with_arc(Self::as_ptr(this), |x| { Arc::strong_count(x) }) }
    }

    fn weak_count(this: &Self) -> usize {
        unsafe { with_arc(Self::as_ptr(this), |x| { Arc::weak_count(x) }) }
    }

    /// # Safety
    ///
    /// It is OK to call this method on any non-null pointer from an ArcLike
    unsafe fn incr_strong_count(ptr: *const T) {
        let x = Arc::from_raw(ptr);
        let y = x.clone(); // clone causes strong increment
        forget(y); // don't let drop decrement
        forget(x); // don't let drop decrement
    }

    unsafe fn decr_strong_count(ptr: *const T) {
        Arc::from_raw(ptr); // implicit drop causes decrement
    }

    // # Safety
    //
    // It is generally a data race to call this method on a DeferredArc or
    // GuardedArc even when the strong_count is one, because there may be
    // other GuardedArcs on other threads
    unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        &mut *(Self::as_ptr(this) as *mut T)
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

    /// # Safety
    /// 
    /// See module documentation
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        assert!(!ptr.is_null());
        Self {
            data: NonZeroUsize::new_unchecked(ptr as usize),
            _marker: PhantomData,
        }
    }

    pub fn get_mut(&mut self) -> Option<&mut T> {
        None
    }

    pub fn as_arc(&self) -> Arc<T> {
        // Safety: safe if the object was constructed safely
        let a = unsafe { Arc::from_raw(self.data.get() as *const T) };
        std::mem::forget(a.clone());
        a
    }

    pub fn into_arc(self) -> Arc<T> {
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

impl<'g, T> ArcInterface<T> for GuardedArc<'g, T> {}

impl<'g, T> ArcLike<T> for GuardedArc<'g, T> {
    
    fn into_usize(this: Self) -> usize {
        let GuardedArc { data, .. } = this;
        data.get()
    }

    fn as_ptr(this: &Self) -> *const T {
        this.data.get() as *const T
    }

    unsafe fn from_usize(data: usize) -> Self {
        debug_assert_ne!(data & ptr_mask::<T>(), 0);
        debug_assert_eq!(data & tag_mask::<T>(), 0);
        Self {
            data: NonZeroUsize::new_unchecked(data),
            _marker: PhantomData
        }
    }
    
}

unsafe impl<'g, T> NotOwning for GuardedArc<'g, T> {}
unsafe impl<'g, T> NotNull for GuardedArc<'g, T> {}



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
        let p = Self::as_ptr(&self);
        std::mem::forget(self);
        p
    }

    /// # Safety
    /// 
    /// The pointer must come from a live arc, be synchronized
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        assert!(!ptr.is_null());
        Self {
            data: NonZeroUsize::new_unchecked(ptr as usize),
            _marker: PhantomData,
        }
    }

   
    pub fn get_mut(&self) -> Option<&mut T> {
        None
    }

    pub fn as_arc(&self) -> Arc<T> {
        let ptr = Self::as_ptr(self);
        unsafe { 
            Self::incr_strong_count(ptr);
            Arc::from_raw(ptr)
        }
    }

    pub fn as_guarded<'g>(&self, _guard: &'g Guard) -> GuardedArc<'g, T> {
        GuardedArc {
            data: self.data,
            _marker: PhantomData,
        }
    }

    pub fn into_arc(self) -> Arc<T> {
        self.as_arc() // increments
        // implicit drop decrements after epoch
    }

    pub fn into_guarded(self, guard: &Guard) -> GuardedArc<T> {
        let data = self.data;
        unsafe { 
            guard.defer_unchecked(move || {
                Self::decr_strong_count(data.get() as *const T)
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
        unsafe { &*Self::as_ptr(&self) }
    }
}

impl<T> std::borrow::Borrow<T> for DeferredArc<T> {
    fn borrow(&self) -> &T {
        unsafe { &*Self::as_ptr(&self) }
    }
}

impl<T> Clone for DeferredArc<T> {
    fn clone(&self) -> Self {
        // Safety: always safe to increment, constructors have asserted pointer is OK
        unsafe { Self::incr_strong_count(self.data.get() as *const T) };
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
                    Self::decr_strong_count(data.get() as *const T)
                } 
            ) 
        }
    }
}
impl<T> std::cmp::Eq for DeferredArc<T> {}

impl<T> From<Arc<T>> for DeferredArc<T> {
    fn from(x: Arc<T>) -> DeferredArc<T> {
        unsafe { Self::from_raw(Arc::into_raw(x)) }
    }
}

impl<T> From<UniqueArc<T>> for DeferredArc<T> {
    fn from(x: UniqueArc<T>) -> DeferredArc<T> {
        unsafe { Self::from_raw(UniqueArc::into_raw(x)) }
    }
}

impl<'g, T> From<GuardedArc<'g, T>> for DeferredArc<T> {
    fn from(x: GuardedArc<T>) -> Self {
        unsafe {
            GuardedArc::incr_strong_count(GuardedArc::as_ptr(&x));
            Self::from_raw(GuardedArc::into_raw(x)) 
        }
    }
}


impl<T> std::cmp::PartialEq<DeferredArc<T>> for DeferredArc<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
} 

unsafe impl<T: Send + Sync> Send for DeferredArc<T> {}
unsafe impl<T: Send + Sync> Sync for DeferredArc<T> {}

impl<T> ArcInterface<T> for DeferredArc<T> {}


impl<'g, T> ArcLike<T> for DeferredArc<T> {

    fn into_usize(this: Self) -> usize {
        let DeferredArc { data, .. } = this;
        data.get()
    }

    fn as_ptr(this: &Self) -> *const T{
        this.data.get() as *const T        
    }

    unsafe fn from_usize(data: usize) -> Self {
        debug_assert_ne!(data & ptr_mask::<T>(), 0);
        debug_assert_eq!(data & tag_mask::<T>(), 0);
        Self {
            data: NonZeroUsize::new_unchecked(data),
            _marker: PhantomData,
        }
    }

}

unsafe impl<T> Owning for DeferredArc<T> {}
unsafe impl<T> NotNull for DeferredArc<T> {}



pub trait Deferred {}
impl<T> Deferred for DeferredArc<T> {}
impl<T: Deferred> Deferred for Option<T> {}
impl<T: Deferred> Deferred for (T, usize) {}


pub struct Loan<'g, T> {
    data: usize,
    _marker: PhantomData<&'g T>
}

impl<'g, T> Loan<'g, T> {

    // In release mode this class could do nothing, depends if we find good
    // reasons to take the slow paths
    
    pub fn new() -> Self {
        Self {
            data: 0,
            _marker: PhantomData,
        }
    }

    pub fn refinance
        <Current: ArcLike<T> + Deferred, New: ArcLike<T> + NotOwning>
        (&mut self, current: &mut Current, new: New, guard: &'g Guard) 
        -> ()
    {
        let new = New::into_usize(new);
        let mut tmp = unsafe { Current::from_usize(new) };
        std::mem::swap(current, &mut tmp);
        let new = new & ptr_mask::<T>();
        let old = Current::into_usize(tmp) & ptr_mask::<T>();
        if (self.data != 0) && ((self.data ^ old) != 0) {
            // wrong pointer!
            let ptr = self.data as *const T;
            unsafe {
                DeferredArc::incr_strong_count(ptr);
                guard.defer_unchecked(move || { 
                    DeferredArc::decr_strong_count(ptr)
                });
            }
        }
        self.data = new;
    }

    pub fn repay<New: ArcLike<T> + Deferred>
        (&mut self, new: New, guard: &'g Guard) 
    {
        let new = New::into_usize(new) & ptr_mask::<T>();
        if new != 0 {
            if new == self.data {
                // dropping the deferred arc repays the original loan
                // (possibly through a chain of swaps)
                self.data = 0;
            } else {
                let ptr = self.data as *const T;
                unsafe {
                    guard.defer_unchecked(move || {
                        DeferredArc::decr_strong_count(ptr)
                    });
                }
            }
        }
    }

}

impl<'g, T> Drop for Loan<'g, T> {
    fn drop(&mut self) {
        let Loan { data, ..} = *self;
        match data {
            0 => (), 
            data => unsafe { DeferredArc::incr_strong_count((data & ptr_mask::<T>()) as *const T) }
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

impl<T> DerefMut for UniqueArc<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.as_mut()
    }
}

impl<T: Clone> From<Arc<T>> for UniqueArc<T> {
    fn from(value: Arc<T>) -> Self {
        let mut value = value;
        Arc::make_mut(&mut value);
        Self { data: value,}
    }
}

impl<T> ArcInterface<T> for UniqueArc<T> {}


impl<T> ArcLike<T> for UniqueArc<T> {

    fn into_usize(this: Self) -> usize {
        Arc::into_usize(this.data)
    }

    fn as_ptr(this: &Self) -> *const T{
        Arc::as_ptr(&this.data)
    }

    unsafe fn from_usize(data: usize) -> Self {
        Self { data: Arc::from_usize(data), }
    }

}

unsafe impl<T> Owning for UniqueArc<T> {}
unsafe impl<T> NotNull for UniqueArc<T> {}







pub struct CompareAndSetError<T, U> {
    pub current: T,
    pub new: U,
}


pub struct AtomicArc<T> {
    data: AtomicUsize,
    _marker: PhantomData<Arc<T>>,
}

impl<T> AtomicArc<T> {

    pub fn new<U: ArcLike<T> + Owning + NotNull>(arc: U) -> Self {
        Self {
            data: AtomicUsize::new(U::into_usize(arc)),
            _marker: PhantomData,
        }
    }

    pub fn into_inner(self) -> (Arc<T>, usize) {
        // Safety: We are being moved so threads must have synchronized with us 
        unsafe { ArcLike::from_usize(self.data.into_inner()) }
    }

    pub unsafe fn load<'g>
        (&self, order: Ordering, _guard: &'g Guard) 
        -> (GuardedArc<'g, T>, usize) 
    {
        ArcLike::from_usize(self.data.load(order)) 
    }

    pub unsafe fn load_consume<'g>
        (&self, _guard: &'g Guard) 
        -> (GuardedArc<'g, T>, usize)
    {
        ArcLike::from_usize(self.data.load_consume())
    }

    pub unsafe fn store
        <'g, N: ArcLike<T> + Owning + NotNull>
        (&self, new: N, order: Ordering, guard: &'g Guard)
        -> ()
    {
        let order = match order {
            Ordering::Release => Ordering::AcqRel,
            order => order
        };
        let (old, ..) = self.swap(new, order);
        old.into_guarded(guard);
    }

    pub unsafe fn swap
        <N: ArcLike<T> + Owning + NotNull>
        (&self, new: N, order: Ordering) 
        -> (DeferredArc<T>, usize) 
    {
        ArcLike::from_usize(self.data.swap(N::into_usize(new), order))
    }
    
    pub unsafe fn compare_and_set
        <'g, C: ArcLike<T> + NotOwning, N: ArcLike<T> + Owning + NotNull, O: CompareAndSetOrdering>
        (&self, current: C, new: N, order: O, _guard: &'g Guard) 
        -> Result<(DeferredArc<T>, usize), CompareAndSetError<(GuardedArc<'g, T>, usize), N>> 
    {
        let new = N::into_usize(new);
        match self.data.compare_exchange
            (C::into_usize(current), new, order.success(), order.failure()) 
        {
            Ok(old) => Ok(ArcLike::from_usize(old)),
            Err(current) => Err(
                CompareAndSetError {
                    current: ArcLike::from_usize(current),
                    new: N::from_usize(new),
                }
            ),
        }        
    }

    pub unsafe fn compare_and_set_weak
        <'g, C: ArcLike<T> + NotOwning, N: ArcLike<T> + Owning + NotNull, O: CompareAndSetOrdering>
        (&self, current: C, new: N, order: O, _guard: &'g Guard) 
        -> Result<(DeferredArc<T>, usize), CompareAndSetError<(GuardedArc<'g, T>, usize), N>> 
    {
        let new = N::into_usize(new);
        match self.data.compare_exchange_weak
            (C::into_usize(current), new, order.success(), order.failure()) 
        {
            Ok(old) => Ok(ArcLike::from_usize(old)),
            Err(current) => Err(
                CompareAndSetError {
                    current: ArcLike::from_usize(current),
                    new: N::from_usize(new),
                }
            ),
        }        
    }

    pub unsafe fn fetch_and<'g>
        (&self, tag: usize, order: Ordering, _guard: &'g Guard) 
        -> (GuardedArc<'g, T>, usize) 
    {
        ArcLike::from_usize(self.data.fetch_and(tag | !tag_mask::<T>(), order))
    }

    pub unsafe fn fetch_or<'g>
        (&self, tag: usize, order: Ordering, _guard: &'g Guard)
        -> (GuardedArc<'g, T>, usize) 
    {
        ArcLike::from_usize(self.data.fetch_or(tag & tag_mask::<T>(), order))
    }

    pub unsafe fn fetch_xor<'g>
        (&self, tag: usize, order: Ordering, _guard: &'g Guard) 
        -> (GuardedArc<'g, T>, usize) 
    {
        ArcLike::from_usize(self.data.fetch_xor(tag & tag_mask::<T>(), order))
    }

}

impl<T: Default> Default for AtomicArc<T> {
    fn default() -> Self {
        AtomicArc::new(Arc::default())
    }
}

impl<T> From<Arc<T>> for AtomicArc<T> {
    fn from(x: Arc<T>) -> Self {
        Self {
            data: AtomicUsize::new(Arc::into_usize(x)),
            _marker: PhantomData,
        }
    }
}














pub struct AtomicOptionArc<T> {
    data: AtomicUsize,
    _marker: PhantomData<Option<Arc<T>>>,
}

impl<T> AtomicOptionArc<T> {

    pub fn new(value: T) -> Self {
        Self {
            data: AtomicUsize::new(ArcLike::into_usize(Arc::new(value))),
            _marker: PhantomData,
        }
    }

    pub fn into_inner(self) -> (Option<Arc<T>>, usize) {
        // Safety: self is being moved so nobody is looking at us
        unsafe { ArcLike::from_usize(self.data.into_inner()) }
    }

    pub unsafe fn load
        <'g>
        (&self, order: Ordering, _guard: &'g Guard) 
        -> (Option<GuardedArc<T>>, usize) 
    {
        ArcLike::from_usize(self.data.load(order))
    }

    pub unsafe fn load_consume
        <'g>
        (&self, _guard: &'g Guard) 
        -> (Option<GuardedArc<T>>, usize)
    {
        ArcLike::from_usize(self.data.load_consume())
    }

    pub unsafe fn store
        <'g, New: ArcLike<T> + Owning>
        (&self, new: New, order: Ordering, guard: &Guard) 
        -> ()
    {
        // Safety: best guess
        let order = match order {
            Ordering::Release => Ordering::AcqRel,
            order => order,
        };
        let (old, ..) = self.swap(new, order);
        old.map(|x| { x.into_guarded(guard) });
    }

    pub unsafe fn 
        swap
        <'g, New: ArcLike<T> + Owning>(&self, new: New, order: Ordering) 
        -> (Option<DeferredArc<T>>, usize) 
    {
        ArcLike::from_usize(self.data.swap(New::into_usize(new), order))
    }

    pub unsafe fn compare_and_set
        <'g, Current: ArcLike<T> + NotOwning, New: ArcLike<T> + Owning, Order: CompareAndSetOrdering>
        (&self, current: Current, new: New, order: Order, _guard: &'g Guard) 
        -> Result<(Option<DeferredArc<T>>, usize), CompareAndSetError<(Option<GuardedArc<'g, T>>, usize), New>> 
    {
        let new = New::into_usize(new);
        match self.data.compare_exchange
            (Current::into_usize(current), new, order.success(), order.failure()) 
        {
            Ok(old) => Ok(ArcLike::from_usize(old)),
            Err(current) => Err(
                CompareAndSetError {
                    current: ArcLike::from_usize(current),
                    new: New::from_usize(new),
                }
            ),
        }        
    }

    pub unsafe fn compare_and_set_weak
        <'g, Current: ArcLike<T> + NotOwning, New: ArcLike<T> + Owning, Order: CompareAndSetOrdering>
        (&self, current: Current, new: New, order: Order, _guard: &'g Guard) 
        -> Result<(Option<DeferredArc<T>>, usize), CompareAndSetError<(Option<GuardedArc<'g, T>>, usize), New>> 
    {
        let new = New::into_usize(new);
        match self.data.compare_exchange_weak
            (Current::into_usize(current), new, order.success(), order.failure()) 
        {
            Ok(old) => Ok(ArcLike::from_usize(old)),
            Err(current) => Err(
                CompareAndSetError {
                    current: ArcLike::from_usize(current),
                    new: New::from_usize(new),
                }
            ),
        }        
    }

    pub unsafe fn fetch_and<'g>
        (&self, tag: usize, order: Ordering, _guard: &'g Guard) 
        -> (Option<GuardedArc<'g, T>>, usize) 
    {
        debug_assert_eq!(tag & ptr_mask::<T>(), 0);
        ArcLike::from_usize(self.data.fetch_and(tag | ptr_mask::<T>(), order))
    }

    pub unsafe fn fetch_or<'g>
        (&self, tag: usize, order: Ordering, _guard: &'g Guard) 
        -> (Option<GuardedArc<'g, T>>, usize)
    {
        debug_assert_eq!(tag & ptr_mask::<T>(), 0);
        ArcLike::from_usize(self.data.fetch_or(tag, order))
    }

    pub unsafe fn fetch_xor<'g>
        (&self, tag: usize, order: Ordering, _guard: &'g Guard)
        -> (Option<GuardedArc<'g, T>>, usize)
    {
        debug_assert_eq!(tag & ptr_mask::<T>(), 0);
        ArcLike::from_usize(self.data.fetch_xor(tag, order))
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



pub struct AtomicArcCell<T> {
    data: AtomicArc<T>,
}

impl<T> AtomicArcCell<T> {

    pub fn new(x: Arc<T>) -> Self {
        AtomicArcCell { 
            data: AtomicArc::from(x) 
        }
    }

    pub fn into_inner(self) -> Arc<T> {
        self.data.into_inner().0
    }

    pub fn get<'g>(&self, guard: &'g Guard) -> GuardedArc<'g, T> {
        unsafe { self.data.load(Ordering::AcqRel, guard) }.0
    }

    pub fn set(&self, new: impl ArcLike<T> + Owning + NotNull, guard: &Guard) {
        unsafe { self.data.store(new, Ordering::Release, guard) }
    }

    pub fn replace(&self, new: impl ArcLike<T> + Owning + NotNull) -> DeferredArc<T> {
        unsafe { self.data.swap(new, Ordering::AcqRel) }.0
    }

}

impl<T: Default> AtomicArcCell<T> {

    pub fn take(&self) -> DeferredArc<T> {
        self.replace(Arc::default())
    }

}




pub struct ArcSwap<T> {
    data: AtomicUsize,
    _marker: PhantomData<Arc<T>>
}

impl<T> ArcSwap<T> {

    pub fn new(x: Arc<T>) -> Self {
        ArcSwap { 
            data: AtomicUsize::new(Arc::into_usize(x)),
            _marker: PhantomData,
        }
    }

    pub fn into_inner(self) -> Arc<T> {
        unsafe { Arc::from_usize(self.data.into_inner()) }
    }

    pub fn swap(&self, new: Arc<T>) -> Arc<T> {
        unsafe { Arc::from_usize(self.data.swap(Arc::into_usize(new), Ordering::AcqRel)) }
    }

}










#[cfg(test)]
mod tests {

    //extern crate crossbeam;

    use super::AtomicOptionArc;
    use std::sync::Arc;
    use crossbeam::epoch::{Guard};
    use std::sync::atomic::Ordering;
    use crate::{ArcInterface, ArcLike, DeferredArc, GuardedArc, UniqueArc, Loan};
    
    struct Node<T> {
        next: Option<DeferredArc<Node<T>>>,
        value: T,
    }

    struct Snapshot<T> {
        head: Option<DeferredArc<Node<T>>>,
    }

    struct Stack<T> {
        head: AtomicOptionArc<Node<T>>,
    }

    impl<T> Default for Stack<T> {
        fn default() -> Self {
            Self {
                head: AtomicOptionArc::default(),
            }
        }
    }

    impl<T> Stack<T> {

        fn snapshot<'g>(&self, guard: &'g Guard) -> Snapshot<T> {
            Snapshot { 
                head: unsafe { 
                    self.head.load(Ordering::Acquire, guard).0.map(|x| { 
                        DeferredArc::from(x) // <-- eager increment
                    })
                }
            }
        }

        fn push<'g>(&self, value: T, guard: &'g Guard) {
            let mut new = UniqueArc::new(Node { next: None, value });
            let mut current = unsafe { self.head.load(Ordering::Relaxed, guard) };
            let mut loan = Loan::new();
            loop {
                loan.refinance(&mut new.next, current.0, guard);
                match unsafe {
                    self.head.compare_and_set_weak(current, new, Ordering::Release, guard)
                } {
                    Ok(old) => {
                        loan.repay(old, guard);
                        return
                    },
                    Err(err) => {
                        current = err.current;
                        new = err.new;
                    }
                }
            }
        }

        fn pop<'g>(&self, value: T, guard: &'g Guard) -> Option<T> {
            let mut current = unsafe { self.head.load(Ordering::Acquire, guard) };
            let loan = Loan::new();
            loop {
                match current {
                    (None, _tag) => return None,
                    (Some(arc), _tag) => unsafe {
                        match self.head.compare_and_set_weak(current, arc.next, Ordering::AcqRel, guard) {
                            Ok((arc, _tag)) => {
                                return arc.map(|x| { Arc::from(x) })
                            },
                            Err(err) => { 
                                current = err.current
                            },
                        }
                    }
                }
            }
        }

    }




    /*
    struct IntrusiveStackNode<T> {
        next: *const IntrusiveStackNode<T>,
        value: T,
    }

    impl<T: Clone> Clone for IntrusiveStackNode<T> {
        fn clone(&self) -> Self {
            Self {
                next: self.next,
                value: self.value.clone()
            }
        }
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
    
    impl<T: Clone + Send + Sync> IntrusiveStack<T> {

        fn new() -> Self {
            Self {
                head: AtomicOptionArc::default(),
            }
        }

        fn top<'g>(&self, guard: &'g epoch::Guard) -> Option<&'g T> {
            let old = unsafe { self.head.load(Ordering::Acquire, &guard) };
            old.0.map(|p| { unsafe { &(*GuardedArc::as_ptr(&p)).value } })
        }

        fn push(&self, new: Arc<IntrusiveStackNode<T>>) {
            let mut new : UniqueArc<IntrusiveStackNode<T>> = UniqueArc::from(new);
            let guard = epoch::pin();
            let mut current = unsafe { self.head.load(Ordering::Relaxed, &guard) };
            loop {
                new.next = ArcLike::as_ptr(&current);
                match unsafe {
                    self.head.compare_and_set_weak(current, new, (Ordering::Release, Ordering::Relaxed), &guard) 
                } {
                    Ok((_old, _tag)) => break,
                    Err(err) => { 
                        current = err.current; 
                        new = err.new;
                        continue 
                    },
                }
            }
        }

        fn pop(&self) -> Option<Arc<IntrusiveStackNode<T>>> {
            let guard = epoch::pin();
            let mut current = unsafe { self.head.load(Ordering::Acquire, &guard) };
            loop {
                match current {
                    (None, _tag) => return None,
                    (Some(node), _tag) => {                        
                        match unsafe { 
                            self.head.compare_and_set_weak(current, node.next, (Ordering::Acquire, Ordering::Acquire), &guard) 
                        } {
                            Ok((old, _tag)) => return old.map(|x| { Arc::from(x) }),
                            Err(err) => {
                                current = err.current;
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
        a.push(Arc::new(IntrusiveStackNode::new(8)));
        a.push(Arc::new(IntrusiveStackNode::new(9)));
        a.push(Arc::new(IntrusiveStackNode::new(10)));
        drop(&g);
        println!("{:?}", x);

    }
    */

    #[test]
    fn it_works() {

    }
}




/*

Tagged ptr
// We want the alignment not of T, but of T that is inside an ArcInner, after two AtomicUsize fields
const fn low_bits<T>() -> usize {
    (std::mem::align_of::<T>() - 1) | (std::mem::align_of::<std::sync::atomic::AtomicUsize>() - 1)
}

fn ensure_aligned<T>(ptr: *const T) -> *const T {
    assert_eq!(ptr as usize & low_bits::<T>(), 0, "misaligned pointer");
    ptr
}

fn compose<T>(ptr: *const T, tag: usize) -> usize {
    debug_assert_eq!(ptr as usize & low_bits::<T>(), 0, "misaligned pointer");
    debug_assert_eq!(tag & !low_bits::<T>(), 0, "oversized tag");
    (ptr as usize) | (tag & low_bits::<T>())
}

fn decompose<T>(data: usize) -> (*const T, usize) {
    ((data & !low_bits::<T>()) as *const T, data & low_bits::<T>())
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct TaggedPtr<T> {
    data: usize,
    _marker: PhantomData<*const T>,
}

impl<T> TaggedPtr<T> {

    fn new(ptr: *const T, tag: usize) -> Self {
        Self {
            data: compose(ptr, tag),
            _marker: PhantomData,
        }
    }

}

impl<T> From<*const T> for TaggedPtr<T> {
    fn from(ptr: *const T) -> TaggedPtr<T>{
        TaggedPtr::new(ptr, 0)
    }
}

impl<T> From<TaggedPtr<T>> for *const T {
    fn from(ptr: TaggedPtr<T>) -> Self {
        let (ptr, ..) = decompose(ptr.data);
        ptr
    }
}



struct TaggedNonNullPtr<T> {
    data: NonZeroUsize,
    _marker: PhantomData<*const T>,
}

impl<T> TaggedNonNullPtr<T> {

    fn new(ptr: NonNullConst<T>, tag: usize) -> Self {
        // Safety: roundtripping a non-zero value
        Self {
            data: unsafe { NonZeroUsize::new_unchecked(compose(ptr.as_ptr(), tag)) },
            _marker: PhantomData,
        }
    }

}

impl<T> From<NonNullConst<T>> for TaggedNonNullPtr<T> {
    fn from(ptr: NonNullConst<T>) -> Self {
        Self::new(ptr, 0)
    }
}

impl<T> From<&T> for TaggedNonNullPtr<T> {
    fn from(r: &T) -> Self {
        // Safety: roundtripping a non-zero value
        Self {
            data: unsafe { NonZeroUsize::new_unchecked(r as *const T as usize)},
            _marker: PhantomData,
        }
    }
}

impl<T> std::convert::TryFrom<*const T> for TaggedNonNullPtr<T> {
    type Error = ();
    fn try_from(ptr: *const T) -> Result<Self, Self::Error> {
        // Safety: sane pointer
        unsafe { ptr.as_ref().map_or(Err(()), |x| { Ok(Self::from(x)) }) }
    }
}


*/



/*


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
    /// ```
    /// use carc::NonNullConst;
    /// use std::sync::Arc;
    /// 
    /// let p = unsafe { NonNullConst::new_unchecked(Arc::into_raw(Arc::new(1234))) };
    /// ```
    pub unsafe fn new_unchecked(ptr: *const T) -> Self {
        Self { data: std::ptr::NonNull::new_unchecked(ptr as *mut T), }
    }

    /// Creates a `NonNullConst` if `ptr` is non-null.
    pub fn new(ptr: *const T) -> Option<Self> {
        std::ptr::NonNull::new(ptr as *mut T).map(|x| { Self { data: x } } )
    }


    /// Returns the underlying non-null pointer
    pub fn as_ptr(&self) -> *const T {
        std::ptr::NonNull::as_ptr(self.data)
    }

    /// Dereferences the content
    /// 
    /// # Safety
    /// 
    /// See module-level documentation
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


*/


/*

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


*/