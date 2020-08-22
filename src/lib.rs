extern crate crossbeam;

use std::cmp;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::option::Option;
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

        fn top(&self) -> Option<T> {
            let guard = epoch::pin();
            let old = self.head.load(Ordering::Acquire, &guard);
            unsafe { old.as_ref().map(|old| { old.value.clone() }) }
        }

    }
    
    impl<T: Send + Sync> IntrusiveStack<T> {

        fn new() -> Self {
            Self {
                head: Atomic::new(None),
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
        
        let a = IntrusiveStack::new();
        assert!(a.top().is_none());
        a.push(Arc::new(IntrusiveStackNode::new(7)));
        assert_eq!(a.top().unwrap(), 7);
        assert_eq!(a.pop().unwrap().value, 7);
        assert!(a.top().is_none());

    }
}
