extern crate crossbeam;

use std::cmp;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::option::Option;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use crossbeam::epoch::{Guard, CompareAndSetOrdering};
use crossbeam::atomic::AtomicConsume;
use std::num::NonZeroUsize;

// v3: track ownership, use AtomicUsize

pub trait ProtectedPointer<'g, T> {
    fn as_protected_usize(self) -> usize;
    unsafe fn from_protected_usize(data: usize) -> Self;
}

pub trait OwningPointer<'g, T> {
    fn into_owning_usize(self) -> usize;
    unsafe fn from_owning_usize(data: usize, guard: &'g Guard) -> Self;
}

pub trait ProtectedNonNull<'g, T> {
    fn as_protected_non_zero_usize(self) -> NonZeroUsize;
    unsafe fn from_protected_non_zero_usize(data: NonZeroUsize) -> Self;
}

pub trait OwningNonNull<'g, T> {
    fn into_owning_non_zero_usize(self) -> NonZeroUsize;
    unsafe fn from_owning_non_zero_usize(data: NonZeroUsize, guard: &'g Guard) -> Self;
}

// into_owned_non_zero_usize implies into_owned_usize
// from_owned_usize implies from_owned_non_zero_usize






impl<'g, T> OwningNonNull<'g, T> for Arc<T> {

    fn into_owning_non_zero_usize(self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(Arc::into_raw(self) as usize)}
    }

    unsafe fn from_owning_non_zero_usize(data: NonZeroUsize, _: &'g Guard) -> Self {
        Arc::from_raw(data.get() as *const T)
    }

}

impl<'g, T> OwningPointer<'g, T> for Arc<T> {

    fn into_owning_usize(self) -> usize {
        Arc::into_raw(self) as usize
    }

    unsafe fn from_owning_usize(data: usize, _: &'g Guard) -> Self {
        match data {
            0 => panic!("null"),
            data => Arc::from_raw(data as *const T),
        }
    }

}

impl<'g, T> OwningPointer<'g, T> for Option<Arc<T>> {
    
    fn into_owning_usize(self) -> usize {
        self.map_or(0, |x| { Arc::into_raw(x) as usize })
    }

    unsafe fn from_owning_usize(data: usize, _: &'g Guard) -> Self {
        match data {
            0 => None,
            data => Some(Arc::from_raw(data as *const T)),
        }
    }

}






/// A pointer without shared ownership of T, valid for the current epoch
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProtectedArc<'g, T: 'g> {
    data: NonZeroUsize,
    _marker: PhantomData<(&'g(), *const T)>,
}

impl<'g, T: 'g> ProtectedArc<'g, T> {

    pub fn as_raw(&self) -> *const T {
        self.data.get() as *const T
    }

    /// Dereferences the pointer
    /// 
    /// Returns a pointer valid during the lifetime `'g`
    /// 
    /// # Safety: The pointer may be null, or may be insufficiently synchronized
    pub unsafe fn deref(&self) -> &'g T {
        &*self.as_raw()
    }

    pub unsafe fn as_ref(&self) -> &'g T {
        &*self.as_raw()
    }

    pub unsafe fn as_arc(&self) -> Arc<T> {
        let a = Arc::from_raw(self.data.get() as *const T);
        mem::forget(a.clone());
        a
    }

    pub unsafe fn into_arc(self) -> Arc<T> {
        self.as_arc()
    }

}

impl<'g, T> ProtectedNonNull<'g, T> for ProtectedArc<'g, T> {
    
    fn as_protected_non_zero_usize(self) -> NonZeroUsize {
        self.data
    }

    unsafe fn from_protected_non_zero_usize(data: NonZeroUsize) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }

}





#[derive(Debug)]
pub struct OwnedArc<'g, T> {
    data: NonZeroUsize,
    guard: &'g Guard,
    _marker: PhantomData<T>,
}

impl<'g, T> OwnedArc<'g, T> {

    pub fn new(value: T, guard: &'g Guard) -> Self {
        unsafe { Self::from_owning_non_zero_usize(Arc::new(value).into_owning_non_zero_usize(), guard) }
    }

    pub fn as_shared(&self) -> ProtectedArc<'g, T> {
        ProtectedArc {
            data: self.data,
            _marker: PhantomData,
        }
    }

    pub fn as_arc(&self) -> Arc<T> {
        let x = unsafe { Arc::from_raw(self.data.get() as *const T) };
        mem::forget(x.clone());
        x
    }

    pub fn upgrade(&self) -> Arc<T> {
        self.as_arc()
    }

    pub fn into_arc(self) -> Arc<T> {
        self.as_arc() // increments
        // implicit drop decrements after epoch
    }

    pub fn into_shared(self) -> ProtectedArc<'g, T> {
        self.as_shared()
        // implicit drop decrements after epoch
    }

}

impl<'g, T> Clone for OwnedArc<'g, T> {
    fn clone(&self) -> Self {
        mem::forget(self.as_arc());
        Self {
            data: self.data,
            guard: self.guard,
            _marker: PhantomData,
        }
    }
}

impl<'g, T> Drop for OwnedArc<'g, T> {
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

impl<'g, T> OwningNonNull<'g, T> for OwnedArc<'g, T> {
    
    fn into_owning_non_zero_usize(self) -> NonZeroUsize {
        let x = self.data;
        mem::forget(self);
        x
    }

    unsafe fn from_owning_non_zero_usize(data: NonZeroUsize, guard: &'g Guard) -> Self {
        Self {
            data,
            guard,
            _marker: PhantomData,
        }
    }
}


impl<'g, T> OwningPointer<'g, T> for Option<OwnedArc<'g, T>> {
    
    fn into_owning_usize(self) -> usize {
        self.map_or(0, |x| { 
            let data = x.data;
            mem::forget(x);
            data.get()
        })
    }

    unsafe fn from_owning_usize(data: usize, guard: &'g Guard) -> Self {
        match data {
            0 => None,
            data => Some(OwnedArc {
                data: NonZeroUsize::new_unchecked(data),
                guard,
                _marker: PhantomData,
            }),
        }
    }
}

impl<'g, T> ProtectedPointer<'g, T> for Option<ProtectedArc<'g, T>> {
    
    fn as_protected_usize(self) -> usize {
        self.map_or(0, |x| {
            x.data.get()
        })
    }

    unsafe fn from_protected_usize(data: usize) -> Self {
        match data {
            0 => None,
            data => Some(ProtectedArc {
                data: NonZeroUsize::new_unchecked(data),
                _marker: PhantomData,
            })
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


struct AtomicArc<T> {
    data: AtomicNonZeroUsize,
    _marker: PhantomData<Arc<T>>,
}

impl<T> AtomicArc<T> {

    pub fn new(value: T) -> Self {
        Self {
            data: AtomicNonZeroUsize::new(Arc::new(value).into_owning_non_zero_usize()),
            _marker: PhantomData,
        }
    }

    pub fn into_inner(self) -> Arc<T> {
        unsafe { Arc::from_raw(self.data.into_inner().get() as *const T) }
    }

    pub fn load<'g>(&self, order: Ordering, _: &'g Guard) -> ProtectedArc<'g, T> {
        unsafe { ProtectedArc::from_protected_non_zero_usize(self.data.load(order)) }
    }

    pub fn load_consume<'g>(&self, _guard: &'g Guard) -> ProtectedArc<'g, T> {
        unsafe { ProtectedArc::from_protected_non_zero_usize(self.data.load_consume()) }
    }

    pub fn store<'g, P: OwningNonNull<'g, T>>(&self, new: P, order: Ordering, guard: &'g Guard) {
        let order = match order {
            Ordering::Release => Ordering::AcqRel,
            order => order
        };
        self.swap(new, order, guard);
    }

    pub fn swap<'g, P: OwningNonNull<'g, T>>(&self, new: P, order: Ordering, guard: &'g Guard) -> OwnedArc<'g, T> {
        unsafe { OwnedArc::from_owning_non_zero_usize(self.data.swap(new.into_owning_non_zero_usize(), order), guard) }
    }
    
    pub fn compare_and_set<'g, P: OwningNonNull<'g, T>, O: CompareAndSetOrdering>(
        &self, 
        current: ProtectedArc<'g, T>, 
        new: P, 
        order: O,
        guard: &'g Guard
    ) -> Result<OwnedArc<'g, T>, CompareAndSetError<ProtectedArc<'g, T>, P>> {
        // Safety:
        // If the operation fails, the owned new value is reconstructed and returned to the caller
        // If the operation succeeds, the owned old value is returned to the caller
        let new = new.into_owning_non_zero_usize();
        match self.data.compare_exchange(
            current.as_protected_non_zero_usize(),
            new,
            order.success(),
            order.failure()
        ) {
            Ok(old) => Ok(unsafe {
                mem::forget(new);
                OwnedArc::from_owning_non_zero_usize(old, guard)
            }),
            Err(current) => Err(
                CompareAndSetError {
                    current: unsafe { ProtectedArc::from_protected_non_zero_usize(current) },
                    new: unsafe { P::from_owning_non_zero_usize(new, guard) }
                }
            ),
        }        
    }

    pub fn compare_and_set_weak<'g, P: OwningNonNull<'g, T>, O: CompareAndSetOrdering>(
        &self, 
        current: ProtectedArc<'g, T>, 
        new: P, 
        order: O,
        guard: &'g Guard
    ) -> Result<OwnedArc<'g, T>, CompareAndSetError<ProtectedArc<'g, T>, P>> {
        // Safety:
        // If the operation fails, the owned new value is reconstructed and returned to the caller
        // If the operation succeeds, the owned old value is returned to the caller
        let new = new.into_owning_non_zero_usize();
        match self.data.compare_exchange_weak(
            current.as_protected_non_zero_usize(),
            new,
            order.success(),
            order.failure()
        ) {
            Ok(old) => Ok(unsafe { OwnedArc::from_owning_non_zero_usize(old, guard) }),
            Err(current) => Err(
                CompareAndSetError {
                    current: unsafe { ProtectedArc::from_protected_non_zero_usize(current) },
                    new: unsafe { P::from_owning_non_zero_usize(new, guard) },
                }
            ),
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
            data: AtomicUsize::new(Arc::new(value).into_owning_usize()),
            _marker: PhantomData,
        }
    }

    pub fn load<'g>(&self, order: Ordering, _: &'g Guard) -> Option<ProtectedArc<'g, T>> {
        unsafe { Option::from_protected_usize(self.data.load(order)) }
    }

    pub fn load_consume<'g>(&self, _guard: &'g Guard) -> Option<ProtectedArc<'g, T>> {
        unsafe { Option::from_protected_usize(self.data.load_consume()) }
    }

    pub fn store<'g, P: OwningPointer<'g, T>>(&self, new: P, order: Ordering, guard: &'g Guard) {
        let order = match order {
            Ordering::Release => Ordering::AcqRel,
            order => order
        };
        self.swap(new, order, guard);
    }

    pub fn swap<'g, P: OwningPointer<'g, T>>(&self, new: P, order: Ordering, guard: &'g Guard) -> Option<OwnedArc<'g, T>> {
        unsafe { Option::from_owning_usize(self.data.swap(new.into_owning_usize(), order), guard) }
    }

    pub fn compare_and_set<'g, P: OwningPointer<'g, T>, O: CompareAndSetOrdering>(
        &self, 
        current: Option<ProtectedArc<'g, T>>, 
        new: P, 
        order: O,
        guard: &'g Guard
    ) -> Result<Option<OwnedArc<'g, T>>, CompareAndSetError<Option<ProtectedArc<'g, T>>, P>> {
        // Safety:
        // If the operation fails, the owned new value is reconstructed and returned to the caller
        // If the operation succeeds, the owned old value is returned to the caller
        let new = new.into_owning_usize();
        match self.data.compare_exchange(
            current.as_protected_usize(),
            new,
            order.success(),
            order.failure()
        ) {
            Ok(old) => Ok(unsafe { Option::from_owning_usize(old, guard) }),
            Err(current) => Err(
                CompareAndSetError {
                    current: unsafe { Option::from_protected_usize(current) },
                    new: unsafe { P::from_owning_usize(new, guard) }
                }
            ),
        }        
    }

    pub fn compare_and_set_weak<'g, P: OwningPointer<'g, T>, O: CompareAndSetOrdering>(
        &self, 
        current: Option<ProtectedArc<'g, T>>, 
        new: P, 
        order: O,
        guard: &'g Guard
    ) -> Result<Option<OwnedArc<'g, T>>, CompareAndSetError<Option<ProtectedArc<'g, T>>, P>> {
        // Safety:
        // If the operation fails, the owned new value is reconstructed and returned to the caller
        // If the operation succeeds, the owned old value is returned to the caller
        let new = new.into_owning_usize();
        match self.data.compare_exchange_weak(
            current.as_protected_usize(),
            new,
            order.success(),
            order.failure()
        ) {
            Ok(old) => Ok(unsafe { Option::from_owning_usize(old, guard) }),
            Err(current) => Err(
                CompareAndSetError {
                    current: unsafe { Option::from_protected_usize(current) },
                    new: unsafe { P::from_owning_usize(new, guard) },
                }
            ),
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










/*

v2: track ownership, use std::sync::atomic::AtomicPtr

pub trait OwnedPointer<T> {
    fn into_owned_ptr(self) -> *const T;
    unsafe fn from_owned_ptr(ptr: *const T) -> Self;
}

impl<T> OwnedPointer<T> for Arc<T> {
    fn into_owned_ptr(self) -> *const T {
        let ptr = Arc::into_raw(self);
        mem::forget(self);
        ptr
    }
    unsafe fn from_owned_ptr(ptr: *const T) -> Self {
        Arc::from_raw(ptr)
    }
}

impl<T> OwnedPointer<T> for OwnedArc<T> {
    fn into_owned_ptr(self) -> *const T {
        // self.ptr = Arc::into_raw(this: Self)
        let ptr = self.ptr;
        mem::forget(self);
        ptr
    }

    unsafe fn  from_owned_ptr(ptr: *const T) -> Self {
        OwnedArc::from_raw(ptr)
    }
}


pub struct SharedArc<'g, T> {
    ptr: *const T,
    _marker: PhantomData<&'g ()>,
}

impl<'g, T> From<*const T> for SharedArc<'g, T> {
    fn from(ptr: *const T) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }
}

pub struct OwnedArc<T> {
    ptr: *const T,
    _marker: PhantomData<T>,
}

pub struct AtomicArc<T> {
    ptr: AtomicPtr<T>, // models a *mut T which is a bit of an impedance mismatch
}

impl<T> AtomicArc<T> {

    pub fn new(p: Arc<T>) -> Self {
        Self {
            ptr: AtomicPtr::new(Arc::into_raw(p) as *mut T),
        }
    }

    // get_mut can't be implemented as Arc's binary representation is Nonnull<ArcInner<T>>, of which the T is a field

    pub fn into_inner(self) -> Arc<T> {
        Arc::from_raw(self.ptr.into_inner())
    }
    
    fn load<'g>(&self, order: Ordering, _guard: &'g epoch::Guard) -> SharedArc<'g, T> {
        SharedArc::from(self.ptr.load(order) as *const T)
    }

    fn swap<P: OwnedPointer<T>>(&self, new: P, order: Ordering) -> OwnedArc<T> {
        OwnedArc::from_owned_ptr(self.ptr.swap(new.into_owned_ptr() as *mut T, order))
    }

    fn compare_exchange_weak<'g, P: OwnedPointer<T>>(&self, current: SharedArc<T>, new: P, success: Ordering, failure: Ordering, guard: &'g epoch::Guard) -> Result<OwnedArc<T>, (SharedArc<'g, T>, P)> {
        let new = new.into_owned_ptr();
        match self.ptr.compare_exchange_weak(current.ptr as *mut T, new as *mut T, success, failure) {
            Ok(old) => Ok(OwnedArc::from_owned_ptr(old)),
            Err(current) => Err((SharedArc::from(current as *const T), P::from_owned_ptr(new))),
        }
    }


}*/













/*

// v1: don't attempt to track ownership, use std::sync::atomic::AtomicPtr

pub trait Pointer<T> {
    fn into_ptr(self) -> *const T;
    unsafe fn from_ptr(ptr: *const T) -> Self;
}

impl<T> Pointer<T> for Arc<T> {
    fn into_ptr(self) -> *const T {
        Arc::into_raw(self)
    }
    unsafe fn from_ptr(ptr: *const T) -> Self {
        Arc::from_raw(ptr)
    }
}

impl<T> Pointer<T> for Option<Arc<T>> {

    fn into_ptr(self) -> *const T {
        match self {
            Some(arc) => Arc::into_raw(arc),
            None => std::ptr::null(),
        }
    }

    unsafe fn from_ptr(ptr: *const T) -> Self {
        ptr.as_ref().map(|ptr| { Arc::from_raw(ptr) })
    }

}

// Pointer with manual reference counting
pub struct Shared<'g, T: 'g> {
    ptr: *const T,
    _marker: PhantomData<&'g ()>,
}

impl <'g, T> Clone for Shared<'g, T> {
    fn clone(&self) -> Self {
        Shared { 
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

impl <'g, T> Copy for Shared<'g, T> {}

impl <'g, T> Pointer<T> for Shared<'g, T> {
    
    fn into_ptr(self) -> *const T {
        self.ptr
    }

    unsafe fn from_ptr(ptr: *const T) -> Self {
        Self { 
            ptr,
            _marker: PhantomData,
        }
    }
    
}

impl<'g, T> Shared<'g, T> {

    /*
    pub fn from_arc(arc: Option<Arc<T>>) -> Self {
        Self {
            ptr: arc.map_or(std::ptr::null(), |arc| { Arc::into_raw(arc) }),
            _marker: PhantomData,
        }
    }
    */

    pub unsafe fn as_arc(&self) -> Option<Arc<T>> {
        self.ptr.as_ref().map(|ptr| { Arc::from_raw(ptr) })
    }

    pub unsafe fn into_arc(self) -> Option<Arc<T>> {
        self.as_arc()
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub unsafe fn as_ref(&self) -> Option<&T> {        
        self.ptr.as_ref()
    }

    pub fn increment(&self) -> &Self {
        let tmp = unsafe { self.as_arc() };
        mem::forget(tmp.clone());
        mem::forget(tmp);
        self
    }

    pub unsafe fn deferred_decrement(&self, guard: &'g epoch::Guard) -> &Self {
        assert!(!self.ptr.is_null());
        let p : *const T = self.ptr;
        guard.defer_unchecked(move || { Arc::from_raw(p) });
        self
    }

}

impl<'g, T> cmp::PartialEq<Shared<'g, T>> for Shared<'g, T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<'g, T> cmp::Eq for Shared<'g, T> {}

impl<'g, T> fmt::Debug for Shared<'g, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Shared").field("ptr", &self.ptr).finish()
    }

}

impl<'g, T> From<Arc<T>> for Shared<'g, T> {
    fn from(arc: Arc<T>) -> Self {
        Self {
            ptr: Arc::into_raw(arc),
            _marker: PhantomData,
        }
    } 
}

impl<'g, T> From<Option<Arc<T>>> for Shared<'g, T> {
    fn from(arc: Option<Arc<T>>) -> Self {
        Self {
            ptr: arc.map_or(std::ptr::null(), |arc| { Arc::into_raw(arc) }),
            _marker: PhantomData,
        }
    } 
}

pub struct Owned<T> {
    ptr: *const T,
    _marker: PhantomData<T>,
}

impl<T> Clone for Owned<T> {
    fn clone(&self) -> Self {

    }
}




/// An atomic `Option<Arc<T>>` that can be safely shared between threads
pub struct Atomic<T> {
    ptr: AtomicPtr<T>,
    _marker: PhantomData<Arc<T>>,
}

impl<T: Send + Sync> Atomic<T> {

    // Creates a new atomic option arc
    pub fn new<P: Pointer<T>>(value: P) -> Self {
        Atomic {
            ptr: AtomicPtr::new(value.into_ptr() as *mut T),
            _marker: PhantomData,
        }
    }

    /// Loads a `Shared` from the atomic pointer
    ///
    /// # Examples
    /// 
    /// ```
    /// use std::sync::Arc;
    /// use std::sync::atomic::Ordering;
    /// use crossbeam::epoch;
    ///
    /// let atomic = carc::Atomic::new(Arc::new(1234));
    /// let guard = epoch::pin();
    /// let shared = atomic.load(Ordering::Acquire, &guard);
    /// unsafe { 
    ///     assert_eq!(shared.as_ref().unwrap(), &1234);
    /// }
    /// ```
    pub fn load<'g>(&self, ord: Ordering, _: &'g epoch::Guard) -> Shared<'g, T> {
        unsafe { Shared::from_ptr(self.ptr.load(ord)) }
    }

    /// Swaps the atomic pointer
    /// 
    /// # Examples
    /// 
    /// ```
    /// use std::sync::Arc;
    /// use std::sync::atomic::Ordering;
    /// use crossbeam::epoch;
    /// 
    /// let atomic = carc::Atomic::new(Arc::new(1234));
    /// let guard = epoch::pin();
    /// let old = atomic.swap(Arc::new(5678), Ordering::AcqRel, &guard);
    /// unsafe { 
    ///     old.deferred_decrement(&guard); // drop old value eventually
    ///     assert_eq!(old.as_ref().unwrap(), &1234); // still valid for now
    /// }
    /// ```
    pub fn swap<'g, P: Pointer<T>>(&self, new: P, ord: Ordering, _: &'g epoch::Guard) -> Shared<'g, T> {
        unsafe { Shared::from_ptr(self.ptr.swap(P::into_ptr(new) as *mut T, ord)) }
    }

    /// Stores the atomic pointer
    /// 
    /// Safety: if the old value represents a `Arc` it will be leaked unless action is taken elsewhere
    /// 
    /// # Examples
    /// 
    /// ```
    /// use std::sync::Arc;
    /// use std::sync::atomic::Ordering;
    /// use crossbeam::epoch;
    /// 
    /// let atomic = carc::Atomic::new(Arc::new(1234));
    /// let guard = epoch::pin();
    /// atomic.store(Arc::new(5678), Ordering::Release, &guard); 
    /// // old value is lost and leaked - prefer swap
    /// ```
    pub fn store<'g, P: Pointer<T>>(&self, new: P, ord: Ordering, _: &'g epoch::Guard) {
        self.ptr.store(P::into_ptr(new) as *mut T, ord);
    }

    /// Exchanges the atomic pointer
    ///
    /// # Examples
    /// 
    /// ```
    /// use std::sync::{Arc, atomic::Ordering};
    /// use crossbeam::epoch;
    /// 
    /// let a = carc::Atomic::new(Arc::new(1234));
    /// let guard = epoch::pin();
    /// let old = a.load(Ordering::Acquire, &guard);
    ///  unsafe {
    ///     let new = carc::Shared::from(Arc::new(1 + old.as_ref().unwrap()));
    ///     match a.compare_exchange(old, new, Ordering::AcqRel, Ordering::Relaxed, &guard) {
    ///         Ok(old) => old,
    ///         Err(old) => new
    ///     }.deferred_decrement(&guard); // clean up what wasn't installed
    ///     let new = a.load(Ordering::Acquire, &guard).increment().as_arc();
    ///     assert_eq!(new.unwrap().as_ref(), &1235);
    /// }
    /// ```
    pub fn compare_exchange<'g, P: Pointer<T>, Q: Pointer<T>>(&self, current: P, new: Q, success: Ordering, failure: Ordering, _guard: &'g epoch::Guard) -> Result<Shared<T>, P>{        
        // Safety: all pointers were previously in an Arc or a Shared
        unsafe {
            match self.ptr.compare_exchange(P::into_ptr(current) as *mut T, Q::into_ptr(new) as *mut T, success, failure) {
                Ok(old) => Ok(Shared::from_ptr(old)),
                Err(old) => Err(P::from_ptr(old)),
            }
        }
    }

}




#[cfg(test)]
mod tests {

    extern crate crossbeam;

    use super::{Atomic, Pointer, Shared};
    use std::sync::Arc;
    use crossbeam::epoch;
    use std::sync::atomic::Ordering;

    struct IntrusiveStackNode<T> {
        next: *const IntrusiveStackNode<T>, // Or Option<Arc<IntrusiveStackNode>>?        
        value: T,
    }

    unsafe impl<T: Send> Send for IntrusiveStackNode<T> {}
    unsafe impl<T: Sync> Sync for IntrusiveStackNode<T> {}
    
    impl<T: Default> Default for IntrusiveStackNode<T> {
        fn default() -> Self {
            Self {
                next: std::ptr::null(),
                value: T::default(),
            }
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
        head: Atomic<IntrusiveStackNode<T>>,
    }


    impl<T: Send + Sync + Clone> IntrusiveStack<T> {

        /*
        fn top(&self) -> Option<T> {
            let guard = epoch::pin();
            let old = self.head.load(Ordering::Acquire, &guard);
            unsafe { old.as_ref().map(|old| { old.value.clone() }) }
        }
        */

    }
    
    impl<T: Send + Sync> IntrusiveStack<T> {

        fn new() -> Self {
            Self {
                head: Atomic::new(None),
            }
        }

        fn top<'g>(&self, guard: &'g epoch::Guard) -> Option<&'g T> {
            let old = self.head.load(Ordering::Acquire, &guard);
            let p = old.as_ptr();
            if p.is_null() {
                None
            } else {
                unsafe { Some(&(*p).value) }
            }
        }
        fn push(&self, new: Arc<IntrusiveStackNode<T>>) {
            let new = Shared::from(new);
            let guard = epoch::pin();
            let mut old = self.head.load(Ordering::Relaxed, &guard);
            loop {
                unsafe { (*(new.as_ptr() as *mut IntrusiveStackNode<T>)).next = old.as_ptr() };
                match self.head.compare_exchange(old, new, Ordering::Release, Ordering::Relaxed, &guard) {
                    Ok(_) => break,
                    Err(b) => { old = b; continue },
                }
            }
        }

        fn pop(&self) -> Option<Arc<IntrusiveStackNode<T>>> {
            let guard = epoch::pin();
            let mut old = self.head.load(Ordering::Acquire, &guard);
            loop {
                if old.as_ptr().is_null() {
                    return None;
                }
                let new = unsafe { Shared::from_ptr((*old.as_ptr()).next) };
                match self.head.compare_exchange(old, new, Ordering::Acquire, Ordering::Acquire, &guard) {
                    // we have to increment because the caller can drop the returned arc before the epoch is over
                    // we have to decrement after the epoch because that count is keeping the node alive for other threads trying to pop
                    Ok(a) => return unsafe { a.deferred_decrement(&guard).increment().into_arc() },
                    Err(e) => old = e,
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
*/