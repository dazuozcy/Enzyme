! RUN: if [ %llvmver -ge 13 ]; then %fc -emit-llvm -S -O0 %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -passes=enzyme -o /dev/stdout | %opt -O0 -S -o %t && %fc -O0 %t -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [ %llvmver -ge 13 ]; then %fc -emit-llvm -S -O1 %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -passes=enzyme -o /dev/stdout | %opt -O1 -S -o %t && %fc -O1 %t -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [ %llvmver -ge 13 ]; then %fc -emit-llvm -S -O2 %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -passes=enzyme -o /dev/stdout | %opt -O2 -S -o %t && %fc -O2 %t -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [ %llvmver -ge 13 ]; then %fc -emit-llvm -S -O3 %loadFortran %s -o /dev/stdout | %opt %loadEnzyme -passes=enzyme -o /dev/stdout | %opt -O3 -S -o %t && %fc -O3 %t -o %t1 && %t1 | FileCheck %s; fi

module math
contains
    real function square( x )
        real, intent(in) :: x
        square = x**2
    end function
end module math

program app
    use enzyme, only: enzyme_autodiff
    use math, only: square
    implicit none
    real :: x, dx

    x = 3
    print *, square(x)

    dx = 0
    call enzyme_autodiff(square, x, dx);

    print *, dx
end program app

! CHECK: 9
! CHECK-NEXT: 6