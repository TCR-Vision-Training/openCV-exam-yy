#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <string>
#include <numeric>

using namespace cv;
using namespace std;


// ============================================================
// 第9题：双目立体视觉——视差计算与深度估计
// ============================================================
//
// 主要内容：
// 1. StereoBM 视差计算
// 2. StereoSGBM 视差计算
// 3. 视差 -> 深度
// 4. 四个目标颜色分割
// 5. 目标区域鲁棒视差估计
// 6. 深度误差计算
// 7. 四目标远近排序
// 8. numDisparities / blockSize 参数实验
// 9. 弱纹理区域分析
//
// 深度公式：
//              f * B
//        Z = -----------
//                d
//
// f = 700 pixel
// B = 80 mm
// ============================================================


// ============================================================
// 全局参数
// ============================================================

const double FOCAL_LENGTH = 700.0;     // pixel
const double BASELINE = 80.0;          // mm

const int SGBM_NUM_DISPARITIES = 128;
const int SGBM_BLOCK_SIZE = 5;


// ============================================================
// 1. 视差 -> 深度
// ============================================================

double disparityToDepth(double disparity)
{
    if (disparity <= 0.0)
        return 0.0;

    return FOCAL_LENGTH * BASELINE / disparity;
}


// ============================================================
// 2. CV_16S -> CV_32F
//
// OpenCV StereoBM / StereoSGBM 的结果通常放大了16倍
// 因此需要除以16
// ============================================================

Mat convertDisparity(const Mat& disparity16)
{
    Mat disparity;

    disparity16.convertTo(
        disparity,
        CV_32F,
        1.0 / 16.0
    );

    return disparity;
}


// ============================================================
// 3. 统计有效视差
// ============================================================

void printDisparityRange(
    const Mat& disparity,
    double maxAllowedDisparity)
{
    Mat validMask =
        (disparity > 0) &
        (disparity < maxAllowedDisparity);

    double minDisparity = 0.0;
    double maxDisparity = 0.0;

    minMaxLoc(
        disparity,
        &minDisparity,
        &maxDisparity,
        nullptr,
        nullptr,
        validMask
    );

    cout << fixed << setprecision(2);

    cout << "有效视差范围："
         << minDisparity
         << " ~ "
         << maxDisparity
         << " pixel"
         << endl;
}


// ============================================================
// 4. 视差伪彩色
//
// 视差越大 -> 越近
// ============================================================

Mat disparityColor(const Mat& disparity)
{
    Mat validMask = disparity > 0;

    double minDisparity = 0.0;
    double maxDisparity = 0.0;

    minMaxLoc(
        disparity,
        &minDisparity,
        &maxDisparity,
        nullptr,
        nullptr,
        validMask
    );

    cout << fixed << setprecision(2);

    cout << "有效视差范围："
         << minDisparity
         << " ~ "
         << maxDisparity
         << " pixel"
         << endl;

    if (maxDisparity <= minDisparity)
    {
        return Mat::zeros(
            disparity.size(),
            CV_8UC3
        );
    }

    Mat normalized =
        Mat::zeros(
            disparity.size(),
            CV_8U
        );

    for (int y = 0;
         y < disparity.rows;
         y++)
    {
        for (int x = 0;
             x < disparity.cols;
             x++)
        {
            float d =
                disparity.at<float>(y, x);

            if (d > 0)
            {
                double value =
                    (d - minDisparity)
                    /
                    (maxDisparity - minDisparity)
                    * 255.0;

                value =
                    max(
                        0.0,
                        min(255.0, value)
                    );

                normalized.at<uchar>(y, x) =
                    static_cast<uchar>(value);
            }
        }
    }

    Mat color;

    applyColorMap(
        normalized,
        color,
        COLORMAP_JET
    );

    return color;
}


// ============================================================
// 5. 创建深度图
// ============================================================

Mat createDepthMap(const Mat& disparity)
{
    Mat depth(
        disparity.size(),
        CV_32F,
        Scalar(0)
    );

    for (int y = 0;
         y < disparity.rows;
         y++)
    {
        for (int x = 0;
             x < disparity.cols;
             x++)
        {
            float d =
                disparity.at<float>(y, x);

            if (d > 1.0)
            {
                depth.at<float>(y, x) =
                    static_cast<float>(
                        disparityToDepth(d)
                    );
            }
        }
    }

    return depth;
}


// ============================================================
// 6. 深度伪彩色
//
// 深度小 -> 近
// 深度大 -> 远
// ============================================================

Mat depthColor(const Mat& depth)
{
    Mat validMask = depth > 0;

    double minDepth = 0.0;
    double maxDepth = 0.0;

    minMaxLoc(
        depth,
        &minDepth,
        &maxDepth,
        nullptr,
        nullptr,
        validMask
    );

    if (maxDepth <= minDepth)
    {
        return Mat::zeros(
            depth.size(),
            CV_8UC3
        );
    }

    Mat normalized =
        Mat::zeros(
            depth.size(),
            CV_8U
        );

    for (int y = 0;
         y < depth.rows;
         y++)
    {
        for (int x = 0;
             x < depth.cols;
             x++)
        {
            float z =
                depth.at<float>(y, x);

            if (z > 0)
            {
                double value =
                    (z - minDepth)
                    /
                    (maxDepth - minDepth)
                    * 255.0;

                value =
                    max(
                        0.0,
                        min(255.0, value)
                    );

                normalized.at<uchar>(y, x) =
                    static_cast<uchar>(value);
            }
        }
    }

    Mat color;

    applyColorMap(
        normalized,
        color,
        COLORMAP_JET
    );

    return color;
}


// ============================================================
// 7. HSV 颜色 Mask
//
// HSV 比直接 BGR 阈值更适合颜色目标提取
// ============================================================

Mat createHSVMask(
    const Mat& image,
    const Scalar& lower,
    const Scalar& upper)
{
    Mat hsv;
    Mat mask;

    cvtColor(
        image,
        hsv,
        COLOR_BGR2HSV
    );

    inRange(
        hsv,
        lower,
        upper,
        mask
    );

    return mask;
}


// ============================================================
// 8. Mask 腐蚀
//
// 去除目标边缘
// 减少边界视差跳变的影响
// ============================================================

Mat erodeMask(const Mat& mask)
{
    Mat result;

    Mat kernel =
        getStructuringElement(
            MORPH_ELLIPSE,
            Size(7, 7)
        );

    erode(
        mask,
        result,
        kernel
    );

    return result;
}


// ============================================================
// 9. 获取有效视差
// ============================================================

vector<double> getValidDisparities(
    const Mat& disparity,
    const Mat& mask,
    double maxDisparity)
{
    vector<double> values;

    for (int y = 0;
         y < disparity.rows;
         y++)
    {
        for (int x = 0;
             x < disparity.cols;
             x++)
        {
            if (mask.at<uchar>(y, x) == 0)
                continue;

            double d =
                disparity.at<float>(y, x);

            // 有效视差
            if (d > 2.0 &&
                d < maxDisparity)
            {
                values.push_back(d);
            }
        }
    }

    return values;
}


// ============================================================
// 10. 中位数
// ============================================================

double median(
    vector<double> values)
{
    if (values.empty())
        return 0.0;

    sort(
        values.begin(),
        values.end()
    );

    size_t n =
        values.size();

    if (n % 2 == 1)
    {
        return values[n / 2];
    }

    return
        (
            values[n / 2 - 1]
            +
            values[n / 2]
        )
        /
        2.0;
}


// ============================================================
// 11. MAD
//
// Median Absolute Deviation
//
// MAD = median(|xi - median(x)|)
//
// 用于鲁棒异常值检测
// ============================================================

double calculateMAD(
    const vector<double>& values,
    double med)
{
    if (values.empty())
        return 0.0;

    vector<double> deviations;

    deviations.reserve(
        values.size()
    );

    for (double value : values)
    {
        deviations.push_back(
            abs(value - med)
        );
    }

    return median(
        deviations
    );
}


// ============================================================
// 12. MAD 异常值过滤
// ============================================================

vector<double> filterByMAD(
    const vector<double>& values)
{
    if (values.empty())
        return vector<double>();

    double med =
        median(values);

    double mad =
        calculateMAD(
            values,
            med
        );

    // 如果所有值几乎相同
    if (mad < 0.01)
    {
        return values;
    }

    // 鲁棒标准差
    double robustSigma =
        1.4826 * mad;

    // 3倍鲁棒标准差
    double threshold =
        3.0 * robustSigma;

    vector<double> filtered;

    for (double d : values)
    {
        if (abs(d - med) <= threshold)
        {
            filtered.push_back(d);
        }
    }

    return filtered;
}


// ============================================================
// 13. 目标视差估计
//
// 处理流程：
//
// 原始 Mask
//     ↓
// 腐蚀
//     ↓
// 有效视差
//     ↓
// 初始中位数
//     ↓
// MAD异常值过滤
//     ↓
// 最终中位数
// ============================================================

double getObjectDisparity(
    const Mat& disparity,
    const Mat& originalMask,
    double maxDisparity)
{
    // --------------------------------------------------------
    // 1. 腐蚀 Mask
    // --------------------------------------------------------

    Mat mask =
        erodeMask(
            originalMask
        );

    // --------------------------------------------------------
    // 2. 获取有效视差
    // --------------------------------------------------------

    vector<double> values =
        getValidDisparities(
            disparity,
            mask,
            maxDisparity
        );

    if (values.empty())
    {
        return 0.0;
    }

    // --------------------------------------------------------
    // 3. 初始中位数
    // --------------------------------------------------------

    double firstMedian =
        median(values);

    // --------------------------------------------------------
    // 4. MAD 异常值过滤
    // --------------------------------------------------------

    vector<double> filtered =
        filterByMAD(values);

    if (filtered.empty())
    {
        return firstMedian;
    }

    // --------------------------------------------------------
    // 5. 最终中位数
    // --------------------------------------------------------

    double finalMedian =
        median(filtered);

    return finalMedian;
}


// ============================================================
// 14. ObjectInfo
// ============================================================

struct ObjectInfo
{
    string name;

    double disparity;

    double depth;
};


// ============================================================
// 15. 分析目标
// ============================================================

ObjectInfo analyzeObject(
    const string& name,
    const Mat& disparity,
    const Mat& mask,
    double maxDisparity)
{
    ObjectInfo result;

    result.name =
        name;

    result.disparity =
        getObjectDisparity(
            disparity,
            mask,
            maxDisparity
        );

    result.depth =
        disparityToDepth(
            result.disparity
        );

    cout << "\n--------------------------------------\n";

    cout << "目标："
         << name
         << endl;

    if (result.disparity <= 0)
    {
        cout << "无法获得有效视差！"
             << endl;

        return result;
    }

    cout << fixed
         << setprecision(2);

    cout << "代表视差 d = "
         << result.disparity
         << " pixel"
         << endl;

    cout << "估计深度 Z = "
         << result.depth
         << " mm"
         << endl;

    cout << "估计深度 Z = "
         << result.depth / 1000.0
         << " m"
         << endl;

    return result;
}


// ============================================================
// 16. 打印理论深度
// ============================================================

void printTheoryDepth(
    const string& name,
    double disparity)
{
    double depth =
        disparityToDepth(
            disparity
        );

    cout << "\n"
         << name
         << endl;

    cout << "d = "
         << fixed
         << setprecision(2)
         << disparity
         << " pixel"
         << endl;

    cout << "Z = "
         << depth
         << " mm"
         << endl;

    cout << "Z = "
         << depth / 1000.0
         << " m"
         << endl;
}


// ============================================================
// 17. 误差计算
// ============================================================

double calculateError(
    double actual,
    double reference)
{
    if (reference <= 0)
        return 0.0;

    return
        abs(actual - reference)
        /
        reference
        *
        100.0;
}


// ============================================================
// 18. SGBM 参数实验
//
// 用于讨论：
//
// numDisparities
// blockSize
// ============================================================

void parameterExperiment(
    const Mat& grayLeft,
    const Mat& grayRight)
{
    cout << "\n";
    cout << "============================================\n";
    cout << "numDisparities 与 blockSize 参数实验\n";
    cout << "============================================\n";

    vector<int> disparityValues =
    {
        64,
        96,
        128
    };

    vector<int> blockValues =
    {
        3,
        5,
        9
    };


    for (int numDisp :
         disparityValues)
    {
        for (int block :
             blockValues)
        {
            if (block % 2 == 0)
                continue;

            int channels = 1;

            int P1 =
                8 *
                channels *
                block *
                block;

            int P2 =
                32 *
                channels *
                block *
                block;

            Ptr<StereoSGBM> sgbm =
                StereoSGBM::create(
                    0,
                    numDisp,
                    block
                );

            sgbm->setP1(P1);
            sgbm->setP2(P2);

            sgbm->setPreFilterCap(31);

            sgbm->setUniquenessRatio(8);

            sgbm->setSpeckleWindowSize(100);

            sgbm->setSpeckleRange(2);

            sgbm->setDisp12MaxDiff(1);

            sgbm->setMode(
                StereoSGBM::MODE_SGBM
            );

            Mat disparity16;

            sgbm->compute(
                grayLeft,
                grayRight,
                disparity16
            );

            Mat disparity =
                convertDisparity(
                    disparity16
                );

            Mat validMask =
                disparity > 0;

            double minD = 0;
            double maxD = 0;

            minMaxLoc(
                disparity,
                &minD,
                &maxD,
                nullptr,
                nullptr,
                validMask
            );

            cout << "\n";

            cout << "numDisparities = "
                 << numDisp
                 << "   ";

            cout << "blockSize = "
                 << block
                 << endl;

            cout << "有效视差范围："
                 << fixed
                 << setprecision(2)
                 << minD
                 << " ~ "
                 << maxD
                 << endl;
        }
    }


    cout << "\n参数影响总结：\n";

    cout << "1. numDisparities 太小："
         << "真实视差超出搜索范围时无法正确匹配。"
         << endl;

    cout << "2. numDisparities 增大："
         << "搜索范围扩大，但计算量增加，也可能增加错误匹配。"
         << endl;

    cout << "3. blockSize 较小："
         << "细节丰富，但对噪声和弱纹理更加敏感。"
         << endl;

    cout << "4. blockSize 较大："
         << "匹配更加稳定，但边缘和小目标细节容易丢失。"
         << endl;

    cout << "5. 实际工程中需要在匹配稳定性、"
         << "计算量和空间分辨率之间折中。"
         << endl;
}


// ============================================================
// 19. 主函数
// ============================================================

int main()
{
    cout << "============================================\n";
    cout << "第9题：双目立体视觉——视差计算与深度估计\n";
    cout << "============================================\n";


    // ========================================================
    // 读取图片
    // ========================================================

    string leftPath =
        "../question9/picture9/image91.png";

    string rightPath =
        "../question9/picture9/image92.png";


    Mat left =
        imread(leftPath);

    Mat right =
        imread(rightPath);


    if (left.empty() ||
        right.empty())
    {
        cout << "图片读取失败！"
             << endl;

        cout << "左图："
             << leftPath
             << endl;

        cout << "右图："
             << rightPath
             << endl;

        return -1;
    }


    cout << "\n图片读取成功！"
         << endl;

    cout << "左图大小："
         << left.cols
         << " × "
         << left.rows
         << endl;

    cout << "右图大小："
         << right.cols
         << " × "
         << right.rows
         << endl;


    // ========================================================
    // 灰度
    // ========================================================

    Mat grayLeft;
    Mat grayRight;

    cvtColor(
        left,
        grayLeft,
        COLOR_BGR2GRAY
    );

    cvtColor(
        right,
        grayRight,
        COLOR_BGR2GRAY
    );


    // ========================================================
    // StereoBM
    // ========================================================

    cout << "\n";
    cout << "============================================\n";
    cout << "StereoBM\n";
    cout << "============================================\n";


    int numDisparitiesBM = 128;

    int blockSizeBM = 15;


    Ptr<StereoBM> stereoBM =
        StereoBM::create(
            numDisparitiesBM,
            blockSizeBM
        );


    Mat disparityBM16;


    stereoBM->compute(
        grayLeft,
        grayRight,
        disparityBM16
    );


    Mat disparityBM =
        convertDisparity(
            disparityBM16
        );


    cout << "\nBM视差范围："
         << endl;


    printDisparityRange(
        disparityBM,
        numDisparitiesBM
    );


    Mat disparityBMColor =
        disparityColor(
            disparityBM
        );


    imwrite(
        "output/disparity_BM_color.png",
        disparityBMColor
    );


    // ========================================================
    // StereoSGBM
    // ========================================================

    cout << "\n";
    cout << "============================================\n";
    cout << "StereoSGBM\n";
    cout << "============================================\n";


    int numDisparitiesSGBM =
        SGBM_NUM_DISPARITIES;

    int blockSizeSGBM =
        SGBM_BLOCK_SIZE;


    int channels = 1;


    int P1 =
        8 *
        channels *
        blockSizeSGBM *
        blockSizeSGBM;


    int P2 =
        32 *
        channels *
        blockSizeSGBM *
        blockSizeSGBM;


    Ptr<StereoSGBM> stereoSGBM =
        StereoSGBM::create(
            0,
            numDisparitiesSGBM,
            blockSizeSGBM
        );


    stereoSGBM->setMinDisparity(0);

    stereoSGBM->setNumDisparities(
        numDisparitiesSGBM
    );

    stereoSGBM->setBlockSize(
        blockSizeSGBM
    );

    stereoSGBM->setP1(P1);

    stereoSGBM->setP2(P2);

    stereoSGBM->setPreFilterCap(31);

    stereoSGBM->setUniquenessRatio(8);

    stereoSGBM->setSpeckleWindowSize(100);

    stereoSGBM->setSpeckleRange(2);

    stereoSGBM->setDisp12MaxDiff(1);

    stereoSGBM->setMode(
        StereoSGBM::MODE_SGBM
    );


    // ========================================================
    // SGBM计算
    // ========================================================

    Mat disparitySGBM16;


    stereoSGBM->compute(
        grayLeft,
        grayRight,
        disparitySGBM16
    );


    Mat disparitySGBM =
        convertDisparity(
            disparitySGBM16
        );


    cout << "\nSGBM视差范围："
         << endl;


    printDisparityRange(
        disparitySGBM,
        numDisparitiesSGBM
    );


    Mat disparitySGBMColor =
        disparityColor(
            disparitySGBM
        );


    imwrite(
        "output/disparity_SGBM_color.png",
        disparitySGBMColor
    );


    // ========================================================
    // 深度计算
    // ========================================================

    cout << "\n";
    cout << "============================================\n";
    cout << "深度计算\n";
    cout << "============================================\n";


    cout << "焦距 f = "
         << FOCAL_LENGTH
         << " pixel"
         << endl;


    cout << "基线 B = "
         << BASELINE
         << " mm"
         << endl;


    cout << "公式：Z = f * B / d"
         << endl;


    Mat depthBM =
        createDepthMap(
            disparityBM
        );


    Mat depthSGBM =
        createDepthMap(
            disparitySGBM
        );


    // ========================================================
    // 深度伪彩色
    // ========================================================

    Mat depthBMColor =
        depthColor(
            depthBM
        );


    Mat depthSGBMColor =
        depthColor(
            depthSGBM
        );


    imwrite(
        "output/depth_BM_color.png",
        depthBMColor
    );


    imwrite(
        "output/depth_SGBM_color.png",
        depthSGBMColor
    );


    // ========================================================
    // HSV颜色分割
    // ========================================================

    cout << "\n";
    cout << "============================================\n";
    cout << "HSV颜色目标分割\n";
    cout << "============================================\n";


    // --------------------------------------------------------
    // 红色
    //
    // HSV 红色跨越 0°
    // 因此需要两个区间
    // --------------------------------------------------------

    Mat redMask1 =
        createHSVMask(
            left,
            Scalar(0, 80, 80),
            Scalar(10, 255, 255)
        );


    Mat redMask2 =
        createHSVMask(
            left,
            Scalar(170, 80, 80),
            Scalar(180, 255, 255)
        );


    Mat redMask;


    bitwise_or(
        redMask1,
        redMask2,
        redMask
    );


    // --------------------------------------------------------
    // 绿色
    // --------------------------------------------------------

    Mat greenMask =
        createHSVMask(
            left,
            Scalar(35, 60, 50),
            Scalar(85, 255, 255)
        );


    // --------------------------------------------------------
    // 蓝色
    // --------------------------------------------------------

    Mat blueMask =
        createHSVMask(
            left,
            Scalar(90, 60, 50),
            Scalar(140, 255, 255)
        );


    // --------------------------------------------------------
    // 黄色
    // --------------------------------------------------------

    Mat yellowMask =
        createHSVMask(
            left,
            Scalar(20, 60, 50),
            Scalar(40, 255, 255)
        );


    // ========================================================
    // 形态学处理
    // ========================================================

    Mat morphologyKernel =
        getStructuringElement(
            MORPH_ELLIPSE,
            Size(5, 5)
        );


    morphologyEx(
        redMask,
        redMask,
        MORPH_OPEN,
        morphologyKernel
    );

    morphologyEx(
        redMask,
        redMask,
        MORPH_CLOSE,
        morphologyKernel
    );


    morphologyEx(
        greenMask,
        greenMask,
        MORPH_OPEN,
        morphologyKernel
    );

    morphologyEx(
        greenMask,
        greenMask,
        MORPH_CLOSE,
        morphologyKernel
    );


    morphologyEx(
        blueMask,
        blueMask,
        MORPH_OPEN,
        morphologyKernel
    );

    morphologyEx(
        blueMask,
        blueMask,
        MORPH_CLOSE,
        morphologyKernel
    );


    morphologyEx(
        yellowMask,
        yellowMask,
        MORPH_OPEN,
        morphologyKernel
    );

    morphologyEx(
        yellowMask,
        yellowMask,
        MORPH_CLOSE,
        morphologyKernel
    );


    // ========================================================
    // 保存 Mask
    // ========================================================

    imwrite(
        "output/red_mask.png",
        redMask
    );

    imwrite(
        "output/green_mask.png",
        greenMask
    );

    imwrite(
        "output/blue_mask.png",
        blueMask
    );

    imwrite(
        "output/yellow_mask.png",
        yellowMask
    );


    // ========================================================
    // 四个目标
    // ========================================================

    cout << "\n";
    cout << "============================================\n";
    cout << "四个目标深度估计（SGBM）\n";
    cout << "============================================\n";


    vector<ObjectInfo> objects;


    ObjectInfo red =
        analyzeObject(
            "红色广告牌",
            disparitySGBM,
            redMask,
            numDisparitiesSGBM
        );


    ObjectInfo green =
        analyzeObject(
            "绿色三棱锥",
            disparitySGBM,
            greenMask,
            numDisparitiesSGBM
        );


    ObjectInfo blue =
        analyzeObject(
            "蓝色圆柱",
            disparitySGBM,
            blueMask,
            numDisparitiesSGBM
        );


    ObjectInfo yellow =
        analyzeObject(
            "黄色球体",
            disparitySGBM,
            yellowMask,
            numDisparitiesSGBM
        );


    if (red.disparity > 0)
        objects.push_back(red);


    if (green.disparity > 0)
        objects.push_back(green);


    if (blue.disparity > 0)
        objects.push_back(blue);


    if (yellow.disparity > 0)
        objects.push_back(yellow);


    // ========================================================
    // 远近排序
    //
    // Z 越小 -> 越近
    // d 越大 -> 越近
    // ========================================================

    sort(
        objects.begin(),
        objects.end(),
        [](const ObjectInfo& a,
           const ObjectInfo& b)
        {
            return a.depth < b.depth;
        }
    );


    // ========================================================
    // 输出排序
    // ========================================================

    cout << "\n";
    cout << "============================================\n";
    cout << "四个目标远近排序（近 → 远）\n";
    cout << "============================================\n";


    for (size_t i = 0;
         i < objects.size();
         i++)
    {
        cout << i + 1
             << ". "
             << objects[i].name
             << "   ";


        cout << "d = "
             << fixed
             << setprecision(2)
             << objects[i].disparity
             << " pixel";


        cout << "   Z = "
             << fixed
             << setprecision(2)
             << objects[i].depth / 1000.0
             << " m"
             << endl;
    }


    // ========================================================
    // 理论值
    // ========================================================

    cout << "\n";
    cout << "============================================\n";
    cout << "题目给定视差的理论深度\n";
    cout << "============================================\n";


    double pyramidDisparity =
        30.0;


    double boardDisparity =
        12.0;


    printTheoryDepth(
        "绿色三棱锥",
        pyramidDisparity
    );


    printTheoryDepth(
        "红色广告牌",
        boardDisparity
    );


    // ========================================================
    // 误差检查
    // ========================================================

    cout << "\n";
    cout << "============================================\n";
    cout << "误差检查\n";
    cout << "============================================\n";


    double pyramidReferenceDepth =
        disparityToDepth(
            pyramidDisparity
        );


    double boardReferenceDepth =
        disparityToDepth(
            boardDisparity
        );


    bool pyramidPass = false;

    bool boardPass = false;


    for (const auto& obj :
         objects)
    {
        // ----------------------------------------------------
        // 绿色三棱锥
        // ----------------------------------------------------

        if (obj.name ==
            "绿色三棱锥")
        {
            double error =
                calculateError(
                    obj.depth,
                    pyramidReferenceDepth
                );


            cout << "\n绿色三棱锥："
                 << endl;


            cout << "参考深度："
                 << fixed
                 << setprecision(2)
                 << pyramidReferenceDepth / 1000.0
                 << " m"
                 << endl;


            cout << "实际深度："
                 << obj.depth / 1000.0
                 << " m"
                 << endl;


            cout << "误差："
                 << error
                 << "%"
                 << endl;


            if (error <= 15.0)
            {
                cout << "√ 满足误差 ≤ 15%"
                     << endl;

                pyramidPass = true;
            }
            else
            {
                cout << "× 不满足误差 ≤ 15%"
                     << endl;
            }
        }


        // ----------------------------------------------------
        // 红色广告牌
        // ----------------------------------------------------

        if (obj.name ==
            "红色广告牌")
        {
            double error =
                calculateError(
                    obj.depth,
                    boardReferenceDepth
                );


            cout << "\n红色广告牌："
                 << endl;


            cout << "参考深度："
                 << fixed
                 << setprecision(2)
                 << boardReferenceDepth / 1000.0
                 << " m"
                 << endl;


            cout << "实际深度："
                 << obj.depth / 1000.0
                 << " m"
                 << endl;


            cout << "误差："
                 << error
                 << "%"
                 << endl;


            if (error <= 15.0)
            {
                cout << "√ 满足误差 ≤ 15%"
                     << endl;

                boardPass = true;
            }
            else
            {
                cout << "× 不满足误差 ≤ 15%"
                     << endl;
            }
        }
    }


    // ========================================================
    // 总体误差判断
    // ========================================================

    cout << "\n";
    cout << "============================================\n";
    cout << "误差要求总结\n";
    cout << "============================================\n";


    if (pyramidPass)
    {
        cout << "绿色三棱锥：√"
             << endl;
    }
    else
    {
        cout << "绿色三棱锥：×"
             << endl;
    }


    if (boardPass)
    {
        cout << "红色广告牌：√"
             << endl;
    }
    else
    {
        cout << "红色广告牌：×"
             << endl;
    }


    if (pyramidPass &&
        boardPass)
    {
        cout << "\n√ 两个目标均满足误差 ≤ 15%"
             << endl;
    }
    else
    {
        cout << "\n× 当前参数下未完全满足误差 ≤ 15%"
             << endl;

        cout << "建议进一步调整 StereoSGBM 参数，"
             << "并检查左右图像是否完成立体校正。"
             << endl;
    }


    // ========================================================
    // 参数实验
    // ========================================================

    parameterExperiment(
        grayLeft,
        grayRight
    );


    // ========================================================
    // 弱纹理分析
    // ========================================================

    cout << "\n";
    cout << "============================================\n";
    cout << "弱纹理区域分析\n";
    cout << "============================================\n";


    cout << "\n为什么弱纹理区域视差不可靠？\n";

    cout << "弱纹理区域缺少明显的灰度或颜色变化，"
         << "左右图像中同一区域存在大量相似候选点，"
         << "因此立体匹配难以确定唯一对应关系。"
         << endl;


    cout << "\n工程补救方法：\n";

    cout << "1. 结构光/散斑："
         << "主动投射纹理，增加场景表面可匹配特征。"
         << endl;


    cout << "2. 置信度过滤："
         << "根据匹配代价、左右一致性等指标过滤低置信度视差。"
         << endl;


    cout << "3. Speckle滤波："
         << "去除视差图中面积较小、孤立的错误匹配区域。"
         << endl;


    cout << "4. 增大适当的 blockSize："
         << "提高弱纹理区域匹配稳定性，但可能损失边缘细节。"
         << endl;


    cout << "5. 合理增加 numDisparities："
         << "保证真实视差位于搜索范围内，但过大将增加计算量和错误匹配。"
         << endl;


    // ========================================================
    // 显示
    // ========================================================

    imshow(
        "Left Image",
        left
    );


    imshow(
        "Right Image",
        right
    );


    imshow(
        "StereoBM Disparity",
        disparityBMColor
    );


    imshow(
        "StereoSGBM Disparity",
        disparitySGBMColor
    );


    imshow(
        "SGBM Depth",
        depthSGBMColor
    );


    imshow(
        "Red Mask",
        redMask
    );


    imshow(
        "Green Mask",
        greenMask
    );


    imshow(
        "Blue Mask",
        blueMask
    );


    imshow(
        "Yellow Mask",
        yellowMask
    );


    // ========================================================
    // 输出文件
    // ========================================================

    cout << "\n";
    cout << "============================================\n";
    cout << "处理完成！\n";
    cout << "============================================\n";


    cout << "\n输出文件：\n";


    cout << "1. output/disparity_BM_color.png\n";

    cout << "2. output/disparity_SGBM_color.png\n";

    cout << "3. output/depth_BM_color.png\n";

    cout << "4. output/depth_SGBM_color.png\n";

    cout << "5. output/red_mask.png\n";

    cout << "6. output/green_mask.png\n";

    cout << "7. output/blue_mask.png\n";

    cout << "8. output/yellow_mask.png\n";


    cout << "\n按任意键退出。"
         << endl;


    waitKey(0);

    destroyAllWindows();


    return 0;
}