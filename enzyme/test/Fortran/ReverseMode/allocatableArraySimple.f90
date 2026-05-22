! RUN: if [ %llvmver -ge 13 ]; then %fc -emit-llvm -S -O0 %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -passes=enzyme -o /dev/stdout | %opt -O0 -S -o out.ll && %fc -O0 out.ll -o a.out && ./a.out | FileCheck %s; fi
! RUN: if [ %llvmver -ge 13 ]; then %fc -emit-llvm -S -O1 %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -passes=enzyme -o /dev/stdout | %opt -O1 -S -o out.ll && %fc -O1 out.ll -o a.out && ./a.out | FileCheck %s; fi
! RUN: if [ %llvmver -ge 13 ]; then %fc -emit-llvm -S -O2 %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -passes=enzyme -o /dev/stdout | %opt -O2 -S -o out.ll && %fc -O2 out.ll -o a.out && ./a.out | FileCheck %s; fi
! RUN: if [ %llvmver -ge 13 ]; then %fc -emit-llvm -S -O3 %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -passes=enzyme -o /dev/stdout | %opt -O3 -S -o out.ll && %fc -O3 out.ll -o a.out && ./a.out | FileCheck %s; fi

module AD
    implicit none
    interface
        subroutine selectFirst__enzyme_autodiff(fnc, x, dx, y, dy)
           interface
               subroutine fnc_decal(a, z)
                   real, allocatable, intent(in) :: a(:)
                   real, intent(inout) :: z
               end subroutine
           end interface
           procedure(fnc_decal) :: fnc
           real, allocatable, intent(in) :: x(:)
           real, allocatable, intent(inout) :: dx(:)
           real, intent(inout) :: y
           real, intent(inout) :: dy
        end subroutine
    end interface

    contains

    subroutine selectFirst(x, y)
        real, allocatable, intent(in) :: x(:)
        real, intent(inout) :: y
        y = x(1)
    end subroutine
end module

program app
    use AD
    implicit none
    real, allocatable :: x(:), dx(:)
    real :: y, dy

    allocate(x(3))
    allocate(dx(3))

    x = [2,3,4]
    dx = [0,0,0]
    y = 0
    dy = 1

    call selectFirst__enzyme_autodiff(selectFirst, x, dx, y, dy)

    print *, int(y)
    print *, int(dx(1))
    print *, int(dx(2))
    print *, int(dx(3))
    print *, int(dy)
end program


! CHECK: 2
! CHECK-NEXT: 1
! CHECK-NEXT: 0
! CHECK-NEXT: 0
! CHECK-NEXT: 0