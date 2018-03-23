#include "em.h"
#include "processcc.h"


/*namedWindow("new", WINDOW_NORMAL);
imshow("new", result);
waitKey(0);*/

// To find the largest connected component
//ncc = connectedComponentsWithStats(result, cc, stats, centroids, 8, CV_16U);

//result.convertTo(result, CV_32F);
//cc.convertTo(cc, CV_32F);
//stats.convertTo(stats, CV_32F);
//centroids.convertTo(centroids, CV_32F);
//
//// Browse through the CCs and find the largest CC (assumed as plant root).
//dist = new int[ncc];

//fcc = (float *)stats.data;
//for (int i = 1; i < ncc; i++)
//{
//    if ((fcc[i * 5 + 4] / (rows * cols)) > 0.7 || (fcc[i * 5 + 4] / (rows * cols)) < 0.0003)
//    {
//        dist[i] = -1;
//        continue;
//    }
//    else
//    {
//        dist[i] = (int)(fcc[i * 5 + 2] * fcc[i * 5 + 2] + fcc[i * 5 + 3] * fcc[i * 5 + 3]) /
//            ((centroids.at<float>(i, 0) - (float)(cols / 2)) * (centroids.at<float>(i, 0) - (float)(cols / 2)) +
//            (centroids.at<float>(i, 1) - (float)(rows / 2)) * (centroids.at<float>(i, 1) - (float)(rows / 2)) + 1);
//    }
//}

//for (int i = 1; i < ncc; i++)
//{
//    if (distmax < dist[i])
//    {
//        distmax = dist[i];
//        ccid = i;
//    }
//}

//if (distmax < 0)
//{
//    cout << "Plant root not found." << endl;
//    exit(0);
//}
//
//fcc = (float *)cc.data;
//for (int i = 0; i < cols; i++)
//{
//    for (int j = 0; j < rows; j++)
//    {
//        if (fcc[j * cols + i] == (float)ccid)
//            fcc[j * cols + i] = 255;
//        else
//            fcc[j * cols + i] = 0;
//    }
//}

//cc = result - cc;
//channels.push_back(result);
//channels.push_back(cc);
//channels.push_back(result);
//
//merge(channels, mresult);

//cc = result - cc;

//for (int i = 0; i < ncc; i++)
//    dist[i] = 0;

//for (int j = 800; j >= 0; j -= 40)
//{
//    rootcnt = 0;
//    for (int i = cols / 4; i < (3 * cols / 4); i++)
//    {
//        if ((fcc[j * cols + i] - fcc[j * cols + i - 1]) > 0)
//        {
//            dist[2 * rootcnt] = i;
//            rootcnt++;
//        }
//        else if ((fcc[j * cols + i] - fcc[j * cols + i - 1]) < 0)
//        {
//            dist[2 * (rootcnt - 1) + 1] = i;
//        }
//    }

//    if (rootcnt == 0)
//        continue;

//    // Get main root width
//    distmax = 0;
//    ccid = 0;
//    for (int i = 0; i < rootcnt; i += 2)
//    {
//        if ((dist[2 * i + 1] - dist[2 * i] + 1) > 200)
//            continue;

//        if ((dist[2 * i + 1] - dist[2 * i] + 1) > distmax)
//        {
//            distmax = (dist[2 * i + 1] - dist[2 * i] + 1);
//            ccid = i;
//            break;
//        }
//    }

//    if (distmax > 0)
//    {
//        if ((dist[2 * rootcnt + 1] - dist[0] + 1) > 200)
//            for (int i = dist[2 * ccid]; i <= dist[2 * ccid + 1]; i++)
//                fcc[j * cols + i] = 0;
//        else
//            for (int i = dist[0]; i <= dist[2 * rootcnt + 1]; i++)
//                fcc[j * cols + i] = 0;

//        cc.convertTo(cc, CV_8U);
//        ncc = connectedComponentsWithStats(cc, result, stats, centroids, 8, CV_16U);
//        result.convertTo(result, CV_32F);
//        cc.convertTo(cc, CV_32F);
//        stats.convertTo(stats, CV_32F);
//        centroids.convertTo(centroids, CV_32F);

//        distmax = 0;
//        ccid = 0;
//        for (int k = 1; k < ncc; k++)
//        {
//            if (stats.at<float>(k, 4) > distmax)
//            {
//                distmax = (int)stats.at<float>(k, 4);
//                ccid = k;
//            }
//        }

//        fcc = (float *)result.data;
//        for (int i = 0; i < cols; i++)
//        {
//            for (int k = 0; k < rows; k++)
//            {
//                if (fcc[k * cols + i] == (float)ccid)
//                    fcc[k * cols + i] = 255;
//                else
//                    fcc[k * cols + i] = 0;
//            }
//        }

//        break;
//    }
//}

//if (rootcnt == 0)
//{
//    cout << "Main root not found." << endl;
//    exit(0);
//}

//imwrite(OutFileCC, result);