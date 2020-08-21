extern crate crossbeam;

use std::cmp;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::sync::{Arc, atomic::{Ordering, AtomicPtr}};
use crossbeam::epoch;


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

    pub fn from_arc(arc: Arc<T>) -> Self {
        Self {
            ptr: Arc::into_raw(arc),
            _marker: PhantomData,
        }
    }

    pub unsafe fn into_arc(self) -> Arc<T> {
        Arc::from_raw(self.ptr)
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn deref(&self) -> &T {
        unsafe {
            &*self.ptr
        }
    }

    pub fn increment(&self) -> &Self {
        let a = unsafe { Arc::from_raw(self.ptr) };
        mem::forget(a.clone());
        mem::forget(a);
        self
    }

    pub fn increment_and_into_arc(self) -> Arc<T> {
        let a = unsafe { Self::into_arc(self) };
        mem::forget(a.clone());
        a
    }

    pub unsafe fn deferred_decrement(&self, guard: &'g epoch::Guard) -> &Self {
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

pub struct Atomic<T> {
    ptr: AtomicPtr<T>,
    _marker: PhantomData<Arc<T>>,
}

impl<T: Send + Sync> Atomic<T> {

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
    ///     assert_eq!(shared.deref(), &1234);
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
    ///     assert_eq!(old.deref(), &1234); // still valid for now
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
    /// let new = carc::Shared::from_arc(Arc::new(1 + unsafe { *old.deref() }));
    /// unsafe { 
    ///     match a.compare_exchange(old, new, Ordering::AcqRel, Ordering::Relaxed, &guard) {
    ///         Ok(old) => old,
    ///         Err(old) => new
    ///     }.deferred_decrement(&guard); // clean up what wasn't installed
    ///     let new = a.load(Ordering::Acquire, &guard).increment_and_into_arc();
    ///     assert_eq!(*new, 1235);
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
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
