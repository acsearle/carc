extern crate crossbeam;

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

// v3: track ownership, use AtomicUsize

pub trait ProtectedPointer<'g, T> {
    fn as_protected_ptr(&self) -> *const T;
    fn as_protected_usize(&self) -> usize;
    unsafe fn from_protected_usize(data: usize) -> Self;
}

pub trait OwningPointer<'g, T> {
    fn into_owning_usize(self) -> usize;
    unsafe fn from_owning_usize(data: usize, guard: &'g Guard) -> Self;
    unsafe fn from_owning_ptr(ptr: *const T, guard:&'g Guard) -> Self;
}

pub trait ProtectedNonNull<'g, T> {
    fn as_protected_non_zero_usize(&self) -> NonZeroUsize;
    unsafe fn from_protected_non_zero_usize(data: NonZeroUsize) -> Self;
}

pub trait OwningNonNull<'g, T> {
    fn into_owning_non_zero_usize(self) -> NonZeroUsize;
    unsafe fn from_owning_non_zero_usize(data: NonZeroUsize, guard: &'g Guard) -> Self;
}

// into_owned_non_zero_usize implies into_owned_usize
// from_owned_usize implies from_owned_non_zero_usize


impl<'g, T> OwningPointer<'g, T> for *const T {
    
    fn into_owning_usize(self) -> usize {
        self as usize
    }

    unsafe fn from_owning_usize(data: usize, _guard: &'g Guard) -> Self {
        data as Self
    }

    unsafe fn from_owning_ptr(ptr: *const T, _guard: &'g Guard) -> Self {
        ptr
    }

}




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

    unsafe fn from_owning_usize(data: usize, _guard: &'g Guard) -> Self {
        match data {
            0 => panic!("null"),
            data => Arc::from_raw(data as *const T),
        }
    }

    unsafe fn from_owning_ptr(ptr: *const T, guard:&'g Guard) -> Self {
        Self::from_owning_usize(ptr as usize, guard)
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

    unsafe fn from_owning_ptr(ptr: *const T, guard: &'g Guard) -> Self {
        Self::from_owning_usize(ptr as usize, guard)
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

impl<'g, T> ProtectedNonNull<'g, T> for ProtectedArc<'g, T> {
    
    fn as_protected_non_zero_usize(&self) -> NonZeroUsize {
        self.data
    }

    unsafe fn from_protected_non_zero_usize(data: NonZeroUsize) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }

}

impl<'g, T> ProtectedPointer<'g, T> for Option<ProtectedArc<'g, T>> {
    
    fn as_protected_ptr(&self) -> *const T {
        match self.as_ref() {
            None => std::ptr::null(),
            Some(x) => x.as_ptr(),
        }
    }

    fn as_protected_usize(&self) -> usize {
        match self.as_ref() {
            None => 0,
            Some(x) => x.as_protected_non_zero_usize().get()
        }
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

    unsafe fn from_owning_ptr(ptr: *const T, guard: &'g Guard) -> Self {
        Self::from_owning_usize(ptr as usize, guard)
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


#[cfg(test)]
mod tests {

    extern crate crossbeam;

    use super::{AtomicOptionArc, ProtectedArc, OwnedArc, OwningPointer, ProtectedPointer, OwningNonNull, ProtectedNonNull};
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
                Arc::get_mut(&mut new).unwrap().next = current.as_protected_ptr();
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
