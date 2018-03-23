
#ifdef __cplusplus
extern "C" {
#endif

	void PerformEM(double *_X, double **_Means, unsigned char *_Y, int _rows, int _cols, int _ncores, int imgreduce, double *_Xfull, int _colsfull, double _fuzzifier, double diffth);

#ifdef __cplusplus
}
#endif
