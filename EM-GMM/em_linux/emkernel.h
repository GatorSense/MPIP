#ifdef __cplusplus
extern "C" {
#endif

    typedef float tfloat;
	void PerformEMcuda(tfloat *_X, double **_Means, unsigned char *_Y, int _rows, int _cols, tfloat _fuzzifier, tfloat diffth);

#ifdef __cplusplus
}
#endif
