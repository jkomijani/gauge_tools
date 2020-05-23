/* JK, 2020: A header file for use from cython. */

void init_genrand(unsigned long s);	/* initializes with a seed */
void init_by_array(unsigned long init_key[], int key_length);
unsigned long genrand_int32(void); 	/* generates a random number on [0,0xffffffff]-interval */
