cdef class site:
    cdef readonly int e[4]
    cpdef int index(self)
    cpdef int ks_eta(self,mu)

cpdef int link_index(site XX, int mu)
cpdef int site_index(site XX)
