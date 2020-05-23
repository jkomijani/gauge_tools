cpdef double rand()                      # uniform distribution in [0,1) or [0,1]
cpdef int    randrange(int n)            #   ...        ...     in range(n) 
cpdef double uniform(double a, double b) #   ..         ...     in [a,b) or [a,b]
cpdef double normal()                    # normal distribution ``N(0,1)`` 
cpdef double complex complex_normal()    # complex variable with independent real and imaginary parts each normal with ``N(0,1/sqrt(2))`` 
