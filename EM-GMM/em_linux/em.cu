#include "em.h"
#include "processcc.h"

using namespace cv;
using namespace std;

static int NUMTHREADS = 0, rows, cols, ndims;
Mat img, imgfull, img2, m, X, Xfull;
MatND hist[3], shist[3], ghist[3], mhist[3];
double **means;
string OutFile, OutFileCC;
string OutDir;
int imgreduce = 0;
int UseGPU = 0;
double fuzzifier;
static double diffth = 100.0;

void display_usage()
{
    cout << "EM Segmentation Program" << endl;
    cout << "Usage: em -[r|g] [-m <number>] [-t <number>] <Path to JPEG image> <path to output directory>" << endl;
    cout << "       -r          Tells em program to reduce the sample size by " << endl;
    cout << "                   selecting the pixels at the center. This switch is" << endl;
    cout << "                   to be used when the number of background pixels are" << endl;
    cout << "                   far higher compared to the foreground pixels." << endl;
    cout << "       -g          Use GPU for computations." << endl;
	cout << "       -m <number> Specifies the fuzzifier for fuzzy EM algorithm." << endl;
	cout << "       -t <number> Specifies the threshold to apply for likelihood map." << endl;
	cout << "examples: em img.jpg d:\\output\\" << endl;
    cout << "          em -r img.jpg d:\\output\\" << endl;
    // cout << "The order of the arguments should not change." << endl;
}

// src : input CV_8UC1 image
// dst : output CV_8UC1 imge
// tol : tolerance, from 0 to 100.
// in  : src image bounds
// out : dst image buonds
void imadjust(const Mat& src, Mat& dst)
{
    float in[3][2] = { { 0.0, 255.0 }, { 0.0, 255.0 }, { 0.0, 255.0 } };
    float out[3][2] = { { 0.0, 255.0 }, { 0.0, 255.0 }, { 0.0, 255.0 } };
    vector<Mat> bgr;
    vector<Mat> dbgr;
    int uhist[3][256], chist[3][256];
    char *ptr;
    int nelements = 0;
    double minv, maxv;

    split(src, bgr);
    nelements = bgr[0].rows * bgr[0].cols;

    for (int i = 0; i < 3; i++)
    {
        dbgr.push_back(bgr[i].clone());
        ptr = dbgr[i].ptr<char>();

        minMaxLoc(dbgr[i], &minv, &maxv);
        
        dbgr[i] = dbgr[i] - minv;
        dbgr[i] = dbgr[i] * (255.0/(maxv - minv));
    }

    merge(dbgr, dst);
    //imwrite(OutDir + "adjusted.jpg", dst);
}

void parse_args(int argc, char *argv[])
{
    int imgidx = 0, outputidx = 0, maxopt = argc - 3, i, slen;
    string s;
    char drive[5], dir[300], fname[100], ext[50], *dname, *bname;

	if (argc < 3 || argc > 9)
	{
		cout << "The program needs at least two arguments - Path to a JPEG image and output directory." << endl;
        display_usage();
        exit(0);
	}

	fuzzifier = -1.0;
	for (i = 1; i <= maxopt; i++)
	{
		if (strcmp(argv[i], "-r") == 0)
			imgreduce = 1;
		if (strcmp(argv[i], "-g") == 0)
			UseGPU = 1;
		if (strcmp(argv[i], "-m") == 0)
		{
			fuzzifier = atof(argv[i + 1]);
			i++;
		}
		if (strcmp(argv[i], "-t") == 0)
		{
			diffth = atof(argv[i + 1]);
			i++;
		}
	}

	m = imread(argv[argc - 2], IMREAD_ANYDEPTH | IMREAD_ANYCOLOR);
	rows = m.rows;
	cols = m.cols;
    
#ifdef __GNUC__
	dname = basename(argv[argc - 2]);
	slen = strlen(dname);
	dname[slen - 4] = '\0';

    s.assign(dname);
    OutDir.assign(argv[argc - 1]);

	if (OutDir[OutDir.length() - 1] != '/')
        OutDir = OutDir + "/";
	
    // CreateDirectory(OutDir.c_str(), NULL);
    mkdir(OutDir.c_str(), S_IRWXU | S_IRWXG);
#else
	_splitpath(argv[argc - 2], drive, dir, fname, ext);
	//dir = dirname(argv[argc - 2]);
	OutDir.assign(argv[argc - 1]);
	s.assign(fname);

	if (OutDir[OutDir.length() - 1] != '\\')
        OutDir = OutDir + "\\";
	
	CreateDirectory(OutDir.c_str(), NULL);
#endif
	
	/*cout << OutDir << endl << "s length = " << s.length() << endl;
	system("pause");*/
	
    OutFile = OutDir + s + "_seg.png";
    OutFileCC = OutDir + s + "_bin.png";

	if (m.data == NULL)
	{
		cout << "Invalid image file specified." << endl;
        exit(0);
	}

    /*Mat m4, m5, m6, m7;
    cvtColor(m, m4, CV_BGR2Lab);

    vector<Mat> v1, v2, v3;

    split(m, v1);
    split(m4, v2);

    v3.push_back(v1[0]);
    v3.push_back(v1[1]);
    v3.push_back(v1[2]);

    merge(v3, m5);
    m5.convertTo(m6, CV_32FC3);
    imadjust(m6, m7);
    m7.convertTo(m, CV_8U);
    namedWindow("new", WINDOW_NORMAL);
    imshow("new", m);
    waitKey(0);*/
    //stretch_contrast();
}

void releasememory()
{
	for (int i = 0; i < 3; i++)
	{
		free(means[i]);

        hist[i].release();
        shist[i].release();
        ghist[i].release();
        mhist[i].release();
	}
	
	free(means);

	img.release();
    imgfull.release();
    m.release();
    img2.release();
	X.release();
    Xfull.release();
}

Mat getreducedimage(Mat img)
{
    Mat rimg;
    int rows = 0, cols = 0;
    int rowstep = 0, colstep = 0;

    rows = img.rows;
    cols = img.cols;
    rowstep = rows / 3;
    colstep = cols / 3;

    rimg = Mat::Mat(img, Rect(colstep, rowstep, 1 * colstep, 1 * rowstep));
    return rimg;
}

// Gets number of possible threads, converts image to double
// extracts the features for em.
void init()
{
    Mat m2, mr, mfull;
    int temp;
	
#ifdef __GNUC__
	NUMTHREADS = sysconf(_SC_NPROCESSORS_ONLN) - 1;
#else
	NUMTHREADS = thread::hardware_concurrency() - 1;
#endif

	NUMTHREADS = (NUMTHREADS == 0) ? 1 : NUMTHREADS;

    if (imgreduce == 1)
    {
        mr = getreducedimage(m);
        m.convertTo(imgfull, CV_64F);
        mr.convertTo(img, CV_64F);
        mr.convertTo(img2, CV_32F);

        imgfull = imgfull / 255.0;
        mfull = imgfull.reshape(3, imgfull.rows * imgfull.cols).reshape(1);
        Xfull = Mat(mfull.rows, mfull.cols, CV_64FC1, mfull.data);
        mfull.copyTo(Xfull);
        Xfull = Xfull.t();
    }
    else
    {
        m.convertTo(img, CV_64F);
        m.convertTo(img2, CV_32F);
    }

	img = img / 255.0;
    img2 = img2 / 255.0;

    temp = img.rows * img.cols;
    m2 = img.reshape(3, temp).reshape(1);
	
	X = Mat(m2.rows, m2.cols, CV_64FC1, m2.data);
	m2.copyTo(X);
	X = X.t();
	
    /*for (int i = 0; i < 10; i++)
        printf("%0.15lf   %0.15lf   %0.15lf\n", X.at<double>(0, i), X.at<double>(1, i), X.at<double>(2, i));*/

    means = (double **) malloc(3 * sizeof(double *));
	for (int i = 0; i < 3; i++)
	{
		means[i] = (double *)malloc(2 * sizeof(double));
	}

	atexit(releasememory);
}

void InitializeMeans()
{
    int channels[] = { 0 };
    int histSize[] = { 200 }, j = 0, n = 0,vecsize;
    float branges[] = { 0.0f, 1.0f }, maxv[2] = { 0.0f, 0.0f }, maxidx[2] = { 0.0f, 0.0f };
    const float* ranges[] = { branges };
    int medfltsize = 5, msize, gsize, startid, found = 0, gaussfiltsize = 11;
    // double gaussflt[5] = { 0.006646032999924, 0.194225554409218, 0.598256825181718, 0.194225554409218, 0.006646032999924 };
    double gaussflt[11] = { 0.002661264665953, 0.013447610713342, 0.047408495762542, 0.116606083672820, 0.200096839755061, 0.239559410860562,
        0.200096839755061, 0.116606083672820, 0.047408495762542, 0.013447610713342, 0.002661264665953 };
	vector<Mat> bgr;
	vector<float> vals, peakvals, peakids;
	//size_t vecsize;
    double gval = 0;

    ndims = 3;
	split(img2, bgr);
	msize = ((medfltsize % 2) == 1) ? (medfltsize - 1) / 2 : medfltsize / 2;
    gsize = ((gaussfiltsize % 2) == 1) ? (gaussfiltsize - 1) / 2 : gaussfiltsize / 2;

	for (int i = 0; i < 3; i++)
	{
        /*cout << bgr[i](cv::Rect(0, 0, 3, 10));
        cout << endl;*/
		calcHist(&bgr[i], 1, channels, Mat(), hist[i], 1, histSize, ranges, true, false);
		hist[i].copyTo(mhist[i]);
		/*cout << hist[i] << endl;*/
        
        for (int j = 0; j < histSize[0]; j++)
        {
            vals.clear();
            for (int k = (j - msize); k <= (j + msize); k++)
            {
            if (k < 0 || k >= histSize[0])
            continue;

            vals.push_back(hist[i].at<float>(k, 0));
            }

            vecsize = vals.size();
            sort(vals.begin(), vals.end());
            mhist[i].at<float>(j, 0) = (vecsize % 2 == 0) ? ((vals[vecsize / 2 - 1] + vals[vecsize / 2]) / 2) : vals[vecsize / 2];
        }

        mhist[i].copyTo(ghist[i]);
        for (j = 0; j < histSize[0]; j++)
        {
            vals.clear();
            for (int k = (j - gsize); k <= (j + gsize); k++)
            {
                if (k < 0)
                    vals.push_back(mhist[i].at<float>(0, 0));
                else if (k >= histSize[0])
                    vals.push_back(mhist[i].at<float>(histSize[0] - 1, 0));
                else
                    vals.push_back(mhist[i].at<float>(k, 0));
            }

            gval = 0;
            for (int k = 0; k < gaussfiltsize; k++)
                gval += gaussflt[k] * vals[k];

            ghist[i].at<float>(j, 0) = (float)gval;
        }

        /*cout << "*****" << endl;
        cout << ghist[i] << endl;*/
        ghist[i].copyTo(shist[i]);

        // To remove unwanted noise/spikes in histogram
		/*for (int j = 0; j < histSize[0]; j++)
		{
			vals.clear();
			for (int k = (j - msize); k <= (j + msize); k++)
			{
				if (k < 0 || k >= histSize[0])
					continue;

				vals.push_back(ghist[i].at<float>(k, 0));
			}

			vecsize = vals.size();
			sort(vals.begin(), vals.end());
			shist[i].at<float>(j, 0) = (vecsize % 2 == 0) ? ((vals[vecsize / 2 - 1] + vals[vecsize / 2]) / 2) : vals[vecsize / 2];
		}*/

        /*cout << "*****" << endl;
		cout << shist[i] << endl;*/
        
        peakvals.clear();
		peakids.clear();
        
        j = 1;
        while (shist[i].at<float>(0, 0) == shist[i].at<float>(j, 0))
            j++;
        if (shist[i].at<float>(0, 0) > shist[i].at<float>(j, 0))
        {
            peakids.push_back(floorf(((float)(j)) / 2.0f));
            peakvals.push_back(shist[i].at<float>(0, 0));
        }

		for (j = 1, n = 0, startid = 0, found = 0; j < (histSize[0] - 1); j++)
		{
			if (shist[i].at<float>(j, 0) > shist[i].at<float>(j + 1, 0) && shist[i].at<float>(j - 1, 0) < shist[i].at<float>(j, 0))
			{
				peakids.push_back((float)j);
				peakvals.push_back(shist[i].at<float>(j, 0));
				n++;
				found = 0;
			}
			else if (shist[i].at<float>(j, 0) > shist[i].at<float>(j + 1, 0) && found == 1)
			{
				peakids.push_back(floorf(((float)(j + startid))/2.0f));
				peakvals.push_back(shist[i].at<float>(j, 0));
				n++;
				found = 0;
			}
            else if (shist[i].at<float>(j - 1, 0) < shist[i].at<float>(j, 0) && shist[i].at<float>(j, 0) == shist[i].at<float>(j + 1, 0) && found == 0)
			{
				startid = j;
				found = 1;
			}
			else if (shist[i].at<float>(j, 0) < shist[i].at<float>(j + 1, 0) && found == 1) // Minimum area in histogram
			{
				found = 0;
			}
            
		}

        j = (histSize[0] - 2);
        while (shist[i].at<float>((histSize[0] - 1), 0) == shist[i].at<float>(j, 0))
            j--;
        if (shist[i].at<float>((histSize[0] - 1), 0) > shist[i].at<float>(j, 0))
        {
            peakids.push_back(floorf(((float)(j + (histSize[0] - 1))) / 2.0f));
            peakvals.push_back(shist[i].at<float>((histSize[0] - 1), 0));
        }
		/*cout << "*****" << endl;
		for (std::vector<float>::const_iterator k = peakids.begin(); k != peakids.end(); ++k)
			std::cout << *k << ' ';
		cout << endl << "*****" << endl;
		for (std::vector<float>::const_iterator k = peakvals.begin(); k != peakvals.end(); ++k)
			std::cout << *k << ' ';
        cout << endl << "*****" << endl;*/

		if (peakids.size() == 1)
		{
			means[i][1] = (peakids[0] / ((float)histSize[0]));
			means[i][0] = 0.05;
		}
		else if (peakids.size() == 0)
		{
			means[i][1] = 0.75f;
			means[i][0] = 0.25f;
		}
		else // If we have at least 2 peaks.
		{
			maxv[0] = 0.0f; maxv[1] = 0.0f;
			maxidx[0] = 0.0f; maxidx[1] = 0.0f;

			for (int j = 0; j < peakids.size(); j++)
			{
				if (peakvals[j] > maxv[1])
				{
					maxv[1] = peakvals[j];
					maxidx[1] = peakids[j];
				}
			}

			for (int j = 0; j < peakids.size(); j++)
			{
				if (peakvals[j] > maxv[0] && peakvals[j] != maxv[1])
				{
					maxv[0] = peakvals[j];
					maxidx[0] = peakids[j];
				}
			}

			means[i][0] = ((double)(maxidx[0] / ((float)histSize[0])));
			means[i][1] = ((double)(maxidx[1] / ((float)histSize[0])));
		}
	}
}

int main(int argc, char *argv[])
{
    Mat result, cc, centroids, stats, scc, mresult, Xfloat;
    uchar *resultptr;
    int ncc = 0, ccid = 0, distmax = -10;
    time_t t = time(NULL);
    vector<Mat> channels;

    cout << std::ctime(&t) << endl;
    cout << "Input file : " << argv[argc - 2] << endl;
    parse_args(argc, argv);

    init();
    
    result = (imgreduce == 0) ? Mat::zeros(1, X.cols, CV_8UC1) : Mat::zeros(1, Xfull.cols, CV_8UC1);
    InitializeMeans();
    resultptr = (uchar *)result.data;

    if (UseGPU == 0)
		PerformEM(X.ptr<double>(), (double **)means, resultptr, X.rows, X.cols, NUMTHREADS, imgreduce, Xfull.ptr<double>(), Xfull.cols, fuzzifier, diffth);
    else
    {
        X.convertTo(Xfloat, CV_32F);
        //X.col(0) *= 0.5;
		PerformEMcuda((tfloat *)Xfloat.ptr<tfloat>(), (double **)means, resultptr, X.rows, X.cols, fuzzifier, diffth);
    }

    result = result.reshape(0, rows);
    imwrite(OutFile, result);
    t = time(NULL);
    cout << std::ctime(&t) << endl;
	//cout << "Means:\n";
	//cout << means[0][0] << " --- " << means[1][0] << " --- " << means[2][0] << "\n";
	//cout << means[0][1] << " --- " << means[1][1] << " --- " << means[2][1] << "\n";
    //system("pause");

    return 0;
}
