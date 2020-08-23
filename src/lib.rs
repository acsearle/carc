extern crate crossbeam;

use std::ptr;
use std::cmp;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ops::Deref;
use std::option::Option;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use crossbeam::epoch::{Guard, CompareAndSetOrdering};
use crossbeam::atomic::AtomicConsume;
use std::num::NonZeroUsize;
use std::ptr::NonNull;

// v3: track ownership, use AtomicUsize

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct NonNullConst<T> {
    data: NonNull<T>,
}

impl<T> NonNullConst<T> {

    fn new(ptr: *const T) -> Option<Self> {
        NonNull::new(ptr as *mut T).map(|x| { Self { data: x } } )
    }

    unsafe fn new_unchecked(ptr: *const T) -> Self {
        Self { data: NonNull::new_unchecked(ptr as *mut T), }
    }

    fn as_ptr(self) -> *const T {
        NonNull::as_ptr(self.data)
    }

    unsafe fn as_ref(&self) -> &T {
        NonNull::as_ref(&self.data)
    }

}



// OptionArc + Arc -> usize
// Arc -> NonZeroUsize
// usize -> Option
// NonZeroUsize -> OptionArc + Arc
//
// OptionArcIn
// OptionArcOut


pub trait ArcLike<'g, T> {
    fn as_non_null(&self) -> NonNullConst<T>;
    fn as_non_zero_usize(&self) -> NonZeroUsize;
    fn into_non_null(self) -> NonNullConst<T>;
    fn into_non_zero_usize(self) -> NonZeroUsize;
    unsafe fn from_non_null(ptr: NonNullConst<T>, guard: &'g Guard) -> Self;
    unsafe fn from_non_zero_usize(data: NonZeroUsize, guard: &'g Guard) -> Self;
}

pub trait OptionArcLike<'g, T> {
    fn as_ptr(&self) -> *const T;
    fn as_usize(&self) -> usize;
    fn into_ptr(self) -> *const T;
    fn into_usize(self) -> usize;
    unsafe fn from_ptr(ptr: *const T, guard: &'g Guard) -> Self;
    unsafe fn from_usize(data: usize, guard: &'g Guard) -> Self;    
}


// into_owned_non_zero_usize implies into_owned_usize
// from_owned_usize implies from_owned_non_zero_usize


impl<'g, T> OptionArcLike<'g, T> for *const T {   

    fn as_ptr(&self) -> *const T {
        *self
    }

    fn as_usize(&self) -> usize {
        *self as usize
    }
    
    fn into_ptr(self) -> *const T {
        self
    }
    
    fn into_usize(self) -> usize {
        self as usize
    }

    unsafe fn from_ptr(ptr: *const T, _guard: &'g Guard) -> Self {
        ptr
    }

    unsafe fn from_usize(data: usize, _guard: &'g Guard) -> Self {
        data as *const T
    }

}

impl<'g, T> ArcLike<'g, T> for Arc<T> {
    
    fn as_non_null(&self) -> NonNullConst<T> {
        unsafe { NonNullConst::new_unchecked(Arc::as_ptr(self)) }
    }

    fn as_non_zero_usize(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(Arc::as_ptr(self) as usize) }
    }

    fn into_non_null(self) -> NonNullConst<T> {
        unsafe { NonNullConst::new_unchecked(Arc::into_raw(self)) }
    }

    fn into_non_zero_usize(self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(Arc::into_raw(self) as usize) }
    }

    unsafe fn from_non_null(ptr: NonNullConst<T>, _guard: &'g Guard) -> Self {
        Arc::from_raw(ptr.as_ptr())
    }

    unsafe fn from_non_zero_usize(data: NonZeroUsize, _guard: &'g Guard) -> Self {
        Arc::from_raw(data.get() as *const T)
    }

}

impl<'g, T> OptionArcLike<'g, T> for Arc<T> {
    
    fn as_ptr(&self) -> *const T {
        Arc::as_ptr(&self)
    }
    
    fn as_usize(&self) -> usize {
        self.as_ptr() as usize
    }

    fn into_ptr(self) -> *const T {
        Arc::into_raw(self)
    }

    fn into_usize(self) -> usize {
        self.into_ptr() as usize
    }

    unsafe fn from_ptr(ptr: *const T, guard: &'g Guard) -> Self {
        assert!(!ptr.is_null());
        Arc::from_raw(ptr)
    }

    unsafe fn from_usize(data: usize, guard: &'g Guard) -> Self {
        Self::from_ptr(data as *const T, guard)
    }

}

impl<'g, T> OptionArcLike<'g, T> for Option<Arc<T>> {
    
    fn as_ptr(&self) -> *const T {
        self.as_ref().map_or(ptr::null(), |x| { Arc::as_ptr(x) })
    }

    fn as_usize(&self) -> usize {
        self.as_ptr() as usize
    }

    fn into_ptr(self) -> *const T {
        self.map_or(ptr::null(), |x| { Arc::into_raw(x) })
    }

    fn into_usize(self) -> usize {
        Self::into_ptr(self) as usize
    }

    unsafe fn from_ptr(ptr: *const T, guard: &'g Guard) -> Self {
        Self::from_usize(ptr as usize, guard)
    }

    unsafe fn from_usize(data: usize, _guard: &'g Guard) -> Self {
        match data {
            0 => None,
            data => Some(Arc::from_raw(data as *const T))
        }
    }

}







/// A pointer without shared ownership of T, valid for the current epoch
#[derive(Debug, Eq, PartialEq)]
pub struct ProtectedArc<'g, T: 'g> {
    data: NonZeroUsize,
    _marker: PhantomData<(&'g(), *const T)>,
}

impl<'g, T: 'g> ProtectedArc<'g, T> {

    pub fn as_ptr(&self) -> *const T {
        self.data.get() as *const T
    }

    pub unsafe fn as_arc(&self) -> Arc<T> {
        let a = Arc::from_raw(self.data.get() as *const T);
        mem::forget(a.clone());
        a
    }

    pub unsafe fn into_arc(self) -> Arc<T> {
        self.as_arc()
    }

    pub fn ptr_eq<'h>(&self, other: ProtectedArc<'h, T>) -> bool {
        self.data == other.data
    }

}

impl<'g, T> Clone for ProtectedArc<'g, T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            _marker: PhantomData,
        }
    }
}

impl<'g, T> Copy for ProtectedArc<'g, T> {}

impl<'g, T> Deref for ProtectedArc<'g, T> {
    
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*(self.data.get() as *const T) }
    }

}



impl<'g, T> ArcLike<'g, T> for ProtectedArc<'g, T> {
    
    fn as_non_null(&self) -> NonNullConst<T> {
        unsafe { NonNullConst::new_unchecked(self.as_non_zero_usize().get() as *const T) }
    }

    fn as_non_zero_usize(&self) -> NonZeroUsize {
        self.data
    }

    fn into_non_null(self) -> NonNullConst<T> {
        self.as_non_null()
    }

    fn into_non_zero_usize(self) -> NonZeroUsize {
        self.as_non_zero_usize()
    }

    unsafe fn from_non_null(ptr: NonNullConst<T>, guard: &'g Guard) -> Self {
        Self::from_non_zero_usize(NonZeroUsize::new_unchecked(ptr.as_ptr() as usize), guard)
    }

    unsafe fn from_non_zero_usize(data: NonZeroUsize, _guard: &'g Guard) -> Self {
        Self {
            data,
            _marker: PhantomData
        }
    }
    
}

impl<'g, T> OptionArcLike<'g, T> for Option<ProtectedArc<'g, T>> {

    fn as_ptr(&self) -> *const T {
        self.as_usize() as *const T
    }

    fn as_usize(&self) -> usize {
        self.map_or(0, |x| { x.data.get() as usize })
    }

    fn into_ptr(self) -> *const T {
        Self::into_usize(self) as *const T
    }

    fn into_usize(self) -> usize {
        self.as_usize()
    }

    unsafe fn from_ptr(ptr: *const T, guard: &'g Guard) -> Self {
        Self::from_usize(ptr as usize, guard)
    }

    unsafe fn from_usize(data: usize, guard: &'g Guard) -> Self {
        match data {
            0 => None,
            data => Some(
                ProtectedArc {
                    data: NonZeroUsize::new_unchecked(data),
                    _marker: PhantomData,
                }
            )
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
        unsafe { Self::from_non_zero_usize(Arc::new(value).into_non_zero_usize(), guard) }
    }

    pub fn as_shared(&self) -> ProtectedArc<'g, T> {
        ProtectedArc {
            data: self.data,
            _marker: PhantomData,
        }
    }

    pub fn as_arc(&self) -> Arc<T> {
        let x : Arc<T> = unsafe { Arc::from_non_zero_usize(self.data, self.guard) };
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

impl<'g, T> ArcLike<'g, T> for OwnedArc<'g, T> {

    fn as_non_null(&self) -> NonNullConst<T> {
        unsafe { NonNullConst::new_unchecked(self.as_non_zero_usize().get() as *const T) }
    }

    fn as_non_zero_usize(&self) -> NonZeroUsize {
        self.data
    }

    fn into_non_null(self) -> NonNullConst<T> {
        unsafe { NonNullConst::new_unchecked(Self::into_non_zero_usize(self).get() as *const T) }
    }

    fn into_non_zero_usize(self) -> NonZeroUsize {
        let data = self.data;
        mem::forget(self);
        data
    }

    unsafe fn from_non_null(ptr: NonNullConst<T>, guard: &'g Guard) -> Self {
        Self::from_non_zero_usize(NonZeroUsize::new_unchecked(ptr.as_ptr() as usize), guard)
    }

    unsafe fn from_non_zero_usize(data: NonZeroUsize, guard: &'g Guard) -> Self {
        Self {
            data,
            guard,
            _marker: PhantomData
        }
    }

}


impl<'g, T> OptionArcLike<'g, T> for Option<OwnedArc<'g, T>> {

    fn as_ptr(&self) -> *const T {
        self.as_usize() as *const T
    }

    fn as_usize(&self) -> usize {
        self.as_ref().map_or(0, |x| { x.data.get() })
    }

    fn into_ptr(self) -> *const T {
        Self::into_usize(self) as *const T
    }

    fn into_usize(self) -> usize {
        self.map_or(0, |x| { let OwnedArc { data: data, .. } = x; data.get() })
    }

    unsafe fn from_ptr(ptr: *const T, guard: &'g Guard) -> Self {
        Self::from_usize(ptr as usize, guard)
    }

    unsafe fn from_usize(data: usize, guard: &'g Guard) -> Self {
        match data {
            0 => None,
            data => Some(
                OwnedArc { 
                    data: NonZeroUsize::new_unchecked(data),
                    guard,
                    _marker: PhantomData
                }
            )
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
            data: AtomicNonZeroUsize::new(Arc::new(value).into_non_zero_usize()),
            _marker: PhantomData,
        }
    }

    pub fn into_inner(self) -> Arc<T> {
        unsafe { Arc::from_raw(self.data.into_inner().get() as *const T) }
    }

    pub fn load<'g>(&self, order: Ordering, guard: &'g Guard) -> ProtectedArc<'g, T> {
        unsafe { ProtectedArc::from_non_zero_usize(self.data.load(order), guard) }
    }

    pub fn load_consume<'g>(&self, guard: &'g Guard) -> ProtectedArc<'g, T> {
        unsafe { ProtectedArc::from_non_zero_usize(self.data.load_consume(), guard) }
    }

    pub fn store<'g, P: ArcLike<'g, T>>(&self, new: P, order: Ordering, guard: &'g Guard) {
        let order = match order {
            Ordering::Release => Ordering::AcqRel,
            order => order
        };
        self.swap(new, order, guard);
    }

    pub fn swap<'g, P: ArcLike<'g, T>>(&self, new: P, order: Ordering, guard: &'g Guard) -> OwnedArc<'g, T> {
        unsafe { OwnedArc::from_non_zero_usize(self.data.swap(new.into_non_zero_usize(), order), guard) }
    }
    
    pub fn compare_and_set<'g, C: ArcLike<'g, T>, N: ArcLike<'g, T>, O: CompareAndSetOrdering>(
        &self, 
        current: C, 
        new: N, 
        order: O,
        guard: &'g Guard
    ) -> Result<OwnedArc<'g, T>, CompareAndSetError<ProtectedArc<'g, T>, N>> {
        // Safety:
        // If the operation fails, the owned new value is reconstructed and returned to the caller
        // If the operation succeeds, the owned old value is returned to the caller
        let new = new.into_non_zero_usize();
        match self.data.compare_exchange(
            current.as_non_zero_usize(),
            new,
            order.success(),
            order.failure()
        ) {
            Ok(old) => Ok(unsafe {
                OwnedArc::from_non_zero_usize(old, guard)
            }),
            Err(current) => Err(
                CompareAndSetError {
                    current: unsafe { ProtectedArc::from_non_zero_usize(current, guard) },
                    new: unsafe { N::from_non_zero_usize(new, guard) }
                }
            ),
        }        
    }

    pub fn compare_and_set_weak<'g, C: ArcLike<'g, T>, N: ArcLike<'g, T>, O: CompareAndSetOrdering>(
        &self, 
        current: C, 
        new: N, 
        order: O,
        guard: &'g Guard
    ) -> Result<OwnedArc<'g, T>, CompareAndSetError<ProtectedArc<'g, T>, N>> {
        // Safety:
        // If the operation fails, the owned new value is reconstructed and returned to the caller
        // If the operation succeeds, the owned old value is returned to the caller
        let new = new.into_non_zero_usize();
        match self.data.compare_exchange_weak(
            current.as_non_zero_usize(),
            new,
            order.success(),
            order.failure()
        ) {
            Ok(old) => Ok(unsafe { OwnedArc::from_non_zero_usize(old, guard) }),
            Err(current) => Err(
                CompareAndSetError {
                    current: unsafe { ProtectedArc::from_non_zero_usize(current, guard) },
                    new: unsafe { N::from_non_zero_usize(new, guard) },
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
            data: AtomicUsize::new(Arc::new(value).into_non_zero_usize().get()),
            _marker: PhantomData,
        }
    }

    pub fn load<'g>(&self, order: Ordering, guard: &'g Guard) -> Option<ProtectedArc<'g, T>> {
        unsafe { Option::from_usize(self.data.load(order), guard) }
    }

    pub fn load_consume<'g>(&self, guard: &'g Guard) -> Option<ProtectedArc<'g, T>> {
        unsafe { Option::from_usize(self.data.load_consume(), guard) }
    }

    pub fn store<'g, N: OptionArcLike<'g, T>>(&self, new: N, order: Ordering, guard: &'g Guard) {
        let order = match order {
            Ordering::Release => Ordering::AcqRel,
            order => order
        };
        self.swap(new, order, guard);
    }

    pub fn swap<'g, N: OptionArcLike<'g, T>>(&self, new: N, order: Ordering, guard: &'g Guard) -> Option<OwnedArc<'g, T>> {
        unsafe { Option::from_usize(self.data.swap(new.into_usize(), order), guard) }
    }

    pub fn compare_and_set<'g, C: OptionArcLike<'g, T>, N: OptionArcLike<'g, T>, O: CompareAndSetOrdering>(
        &self, 
        current: C, 
        new: N, 
        order: O,
        guard: &'g Guard
    ) -> Result<Option<OwnedArc<'g, T>>, CompareAndSetError<Option<ProtectedArc<'g, T>>, N>> {
        // Safety:
        // If the operation fails, the owned new value is reconstructed and returned to the caller
        // If the operation succeeds, the owned old value is returned to the caller
        let new = new.into_usize();
        match self.data.compare_exchange(
            current.as_usize(),
            new,
            order.success(),
            order.failure()
        ) {
            Ok(old) => Ok(unsafe { Option::from_usize(old, guard) }),
            Err(current) => Err(
                CompareAndSetError {
                    current: unsafe { Option::from_usize(current, guard) },
                    new: unsafe { N::from_usize(new, guard) }
                }
            ),
        }        
    }

    pub fn compare_and_set_weak<'g, C: OptionArcLike<'g, T>, N: OptionArcLike<'g, T>, O: CompareAndSetOrdering>(
        &self, 
        current: C, 
        new: N, 
        order: O,
        guard: &'g Guard
    ) -> Result<Option<OwnedArc<'g, T>>, CompareAndSetError<Option<ProtectedArc<'g, T>>, N>> {
        // Safety:
        // If the operation fails, the owned new value is reconstructed and returned to the caller
        // If the operation succeeds, the owned old value is returned to the caller
        let new = new.into_usize();
        match self.data.compare_exchange_weak(
            current.as_usize(),
            new,
            order.success(),
            order.failure()
        ) {
            Ok(old) => Ok(unsafe { Option::from_usize(old, guard) }),
            Err(current) => Err(
                CompareAndSetError {
                    current: unsafe { Option::from_usize(current, guard) },
                    new: unsafe { N::from_usize(new, guard) },
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


#[cfg(test)]
mod tests {

    extern crate crossbeam;

    use super::{AtomicOptionArc, ProtectedArc, OwnedArc, ArcLike, OptionArcLike};
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
                Arc::get_mut(&mut new).unwrap().next = current.as_ptr();
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
                            Ok(a) => return Some(unsafe {p.as_arc()}),
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
