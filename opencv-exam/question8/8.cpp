// -*- coding: utf-8 -*-
/*
 * 第 8 题：双视图图像拼接（实操，不调用 Stitcher 类）—— C++ / OpenCV 实现
 * ---------------------------------------------------------------------------
 * 实现步骤：
 * (1) SIFT（或 ORB 降级）特征提取 + 最近邻匹配（Lowe 比率测试过滤），
 *     用 cv::findHomography(..., RANSAC) 估计右图 -> 左图的单应矩阵 H；
 *     由于本场景近似"平移+增益"，再用匹配位移中位数构造平移模型做稳健校验：
 *     若 RANSAC 结果的透视项过大、或内点数不优于平移模型，则回退到平移模型，
 *     避免特征点稀疏时 RANSAC 拟合出畸形单应（画布异常、出现黑缝等）；
 * (2) 由左图四角与右图四角经 H 投影后的包围盒确定全景画布范围，
 *     将右图 warp 到左图坐标系并完成合成；
 * (3) 接缝处理：
 *     a. 曝光增益补偿：在重叠区用灰度比值（中位数）估计增益 g。
 *        右图约 0.88 倍增益，故 g 应接近 1/0.88 ≈ 1.14，右图乘以 g；
 *     b. 重叠区线性羽化：alpha 从重叠区左缘 1 线性渐变到右缘 0，
 *        result = alpha * left + (1 - alpha) * right_gain；
 *     输出最终全景图 q8_panorama.png，接缝处无明显亮度跳变与重影。
 *
 * 羽化带宽 B（alpha 从 1 过渡到 0 的像素宽度）对鬼影与细节模糊的影响：
 *     - B 过小：增益补偿的残余误差会表现为接缝处明显的亮度跳变（拼缝可见）；
 *     - B 适中：配合增益补偿后过渡平滑、无亮度跳变。对本场景（近似纯平移+增益、
 *       配准精度高），取 B ≈ 重叠区宽度即可，重影几乎不可见；
 *     - B 过大：若配准存在残余误差（视差或单应残差），重叠区内左右两幅图会同时
 *       可见，形成鬼影/重影；宽过渡带等效于空间低通混合，会使接缝附近的高频
 *       细节模糊（双重曝光）。
 *     因此带宽需在“压住亮度跳变”与“避免鬼影、保留细节”之间折中：配准误差小时
 *     可选较大带宽，配准误差大时应减小带宽或改用多频段融合/最佳接缝等更高级方法。
 *
 * 编译（二选一）：
 *   1) g++ -O2 q8_stitch.cpp -o q8_stitch $(pkg-config --cflags --libs opencv4)
 *   2) cmake -S . -B build && cmake --build build   （配合附带的 CMakeLists.txt）
 * 运行：
 *   ./q8_stitch [左图路径] [右图路径] [输出路径]   （参数可省略，默认 q8_left.png /
 *   q8_right.png / q8_panorama.png）
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace cv;
using namespace std;

const string LEFT_PATH = "../question8/picture8/image81.png";
const string RIGHT_PATH = "../question8/picture8/image82.png";
const string OUT_PATH = "q8_panorama.png";

const float RATIO_TEST = 0.75f;   // Lowe 比率测试阈值
const int MIN_MATCHES = 8;        // 最少匹配点数
const int FEATHER_BANDWIDTH = 0;  // 羽化带宽（像素）；0 = 自动取整个重叠区宽度
const bool SHOW_RESULT = true;    // 是否直接弹出窗口显示拼接结果

struct MatchResult {
    Mat H;         // 右图 -> 左图的单应矩阵
    int total = 0;
    int inliers = 0;
};

// 读取图像（转 BGR，丢弃 alpha 通道）
Mat loadImage(const string& path) {
    Mat img = imread(path, IMREAD_COLOR);
    if (img.empty()) {
        throw runtime_error("无法读取图像: " + path);
    }
    return img;
}

// 在 left、right 间提取并匹配特征，返回 H: right -> left
MatchResult detectAndMatch(const Mat& left, const Mat& right) {
    Ptr<Feature2D> detector;
    BFMatcher matcher(NORM_L2);

#if (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 4) || CV_VERSION_MAJOR >= 5
    try {
        detector = SIFT::create();   // OpenCV >= 4.4 主库自带 SIFT
    } catch (const cv::Exception&) { // 非自由算法被禁用时降级 ORB
        detector = ORB::create(4000);
        matcher = BFMatcher(NORM_HAMMING);
    }
#else
    detector = ORB::create(4000);    // 旧版 OpenCV 降级 ORB
    matcher = BFMatcher(NORM_HAMMING);
#endif

    vector<KeyPoint> kp1, kp2;
    Mat des1, des2;
    detector->detectAndCompute(left, noArray(), kp1, des1);
    detector->detectAndCompute(right, noArray(), kp2, des2);

    // 最近邻 + 次近邻，Lowe 比率测试过滤误匹配
    vector<vector<DMatch>> knn;
    matcher.knnMatch(des1, des2, knn, 2);

    vector<DMatch> good;
    for (const auto& pair : knn) {
        if (pair.size() == 2 && pair[0].distance < RATIO_TEST * pair[1].distance) {
            good.push_back(pair[0]);
        }
    }
    if (good.size() < static_cast<size_t>(MIN_MATCHES)) {
        throw runtime_error("匹配点过少: " + to_string(good.size()));
    }

    // src 取右图特征点、dst 取左图特征点 => H 将右图坐标映射到左图坐标
    vector<Point2f> src, dst;
    src.reserve(good.size());
    dst.reserve(good.size());
    for (const auto& m : good) {
        src.push_back(kp2[m.trainIdx].pt);
        dst.push_back(kp1[m.queryIdx].pt);
    }

    Mat mask;
    Mat H = findHomography(src, dst, RANSAC, 3.0, mask);

    // 平移模型稳健校验（本场景近似平移+增益）：
    // 以匹配位移的中位数作为平移估计，若 RANSAC 单应退化则回退平移模型。
    vector<Point2f> offsets(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        offsets[i] = dst[i] - src[i];
    }
    vector<float> ox, oy;
    ox.reserve(offsets.size());
    oy.reserve(offsets.size());
    for (const auto& o : offsets) {
        ox.push_back(o.x);
        oy.push_back(o.y);
    }
    const auto median = [](vector<float>& v) -> float {
        const size_t mid = v.size() / 2;
        nth_element(v.begin(), v.begin() + static_cast<ptrdiff_t>(mid), v.end());
        return v[mid];
    };
    const double tdx = median(ox), tdy = median(oy);
    const Mat Ht = (Mat_<double>(3, 3) << 1, 0, tdx,
                                           0, 1, tdy,
                                           0, 0, 1);

    const auto inlierCount = [&](const Mat& Hh, double th) -> int {
        vector<Point2f> proj;
        perspectiveTransform(src, proj, Hh);
        int cnt = 0;
        for (size_t i = 0; i < proj.size(); ++i) {
            if (norm(proj[i] - dst[i]) < th) ++cnt;
        }
        return cnt;
    };

    int n_h = inlierCount(H, 3.0);
    const int n_t = inlierCount(Ht, 3.0);
    const bool degenerate =
        max(abs(H.at<double>(2, 0)), abs(H.at<double>(2, 1))) > 1e-3 || n_t >= n_h;
    if (degenerate) {
        H = Ht;
        n_h = n_t;
    }

    MatchResult res;
    res.H = H;
    res.total = static_cast<int>(good.size());
    res.inliers = n_h;
    return res;
}

// 在重叠区用灰度比值中位数估计右图需要的增益 g
double estimateGain(const Mat& leftCanvas, const Mat& rightWarped, const Mat& overlapMask) {
    Mat grayL, grayR;
    cvtColor(leftCanvas, grayL, COLOR_BGR2GRAY);
    cvtColor(rightWarped, grayR, COLOR_BGR2GRAY);

    vector<float> ratios;
    ratios.reserve(static_cast<size_t>(countNonZero(overlapMask)));
    for (int y = 0; y < overlapMask.rows; ++y) {
        for (int x = 0; x < overlapMask.cols; ++x) {
            if (overlapMask.at<uchar>(y, x) == 0) continue;
            const float vl = grayL.at<uchar>(y, x);
            const float vr = grayR.at<uchar>(y, x);
            if (vl > 20.f && vr > 20.f) {   // 避开暗部与除零
                ratios.push_back(vl / vr);
            }
        }
    }
    if (ratios.empty()) return 1.0;

    const size_t mid = ratios.size() / 2;
    nth_element(ratios.begin(), ratios.begin() + static_cast<ptrdiff_t>(mid), ratios.end());
    double g = ratios[mid];
    return max(0.2, min(5.0, g));
}

int main(int argc, char** argv) {
    try {
        const string leftPath  = argc > 1 ? argv[1] : LEFT_PATH;
        const string rightPath = argc > 2 ? argv[2] : RIGHT_PATH;
        const string outPath   = argc > 3 ? argv[3] : OUT_PATH;

        const Mat left = loadImage(leftPath);
        const Mat right = loadImage(rightPath);
        const int hL = left.rows, wL = left.cols;
        const int hR = right.rows, wR = right.cols;

        // ---------- (1) 特征提取、匹配、单应估计 ----------
        const MatchResult mr = detectAndMatch(left, right);
        cout << "匹配点: " << mr.total
             << ", 模型内点: " << mr.inliers
             << " (" << 100.0 * mr.inliers / max(mr.total, 1) << "%)" << endl;
        cout << "H (右图 -> 左图):\n" << mr.H << endl;

        // ---------- (2) 画布范围 + warp 右图 ----------
        vector<Point2f> corners = {Point2f(0, 0), Point2f(wR - 1, 0),
                                   Point2f(0, hR - 1), Point2f(wR - 1, hR - 1)};
        vector<Point2f> warpedPts;
        perspectiveTransform(corners, warpedPts, mr.H);

        vector<Point2f> all = {Point2f(0, 0), Point2f(wL - 1, 0),
                               Point2f(0, hL - 1), Point2f(wL - 1, hL - 1)};
        all.insert(all.end(), warpedPts.begin(), warpedPts.end());

        auto minXIt = min_element(all.begin(), all.end(),
                                  [](const Point2f& a, const Point2f& b) { return a.x < b.x; });
        auto maxXIt = max_element(all.begin(), all.end(),
                                  [](const Point2f& a, const Point2f& b) { return a.x < b.x; });
        auto minYIt = min_element(all.begin(), all.end(),
                                  [](const Point2f& a, const Point2f& b) { return a.y < b.y; });
        auto maxYIt = max_element(all.begin(), all.end(),
                                  [](const Point2f& a, const Point2f& b) { return a.y < b.y; });

        const int offX = static_cast<int>(lround(-minXIt->x));
        const int offY = static_cast<int>(lround(-minYIt->y));
        const int canvasW = static_cast<int>(lround(maxXIt->x - minXIt->x + 1));
        const int canvasH = static_cast<int>(lround(maxYIt->y - minYIt->y + 1));

        // M = 平移偏移 * H，把右图直接画到全景画布坐标系
        const Mat T = (Mat_<double>(3, 3) << 1, 0, offX,
                                             0, 1, offY,
                                             0, 0, 1) * mr.H;

        Mat warpedRight, warpedMask;
        warpPerspective(right, warpedRight, T, Size(canvasW, canvasH));
        warpPerspective(Mat::ones(hR, wR, CV_8UC1) * 255, warpedMask, T,
                        Size(canvasW, canvasH), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

        Mat canvas = Mat::zeros(canvasH, canvasW, CV_8UC3);
        left.copyTo(canvas(Rect(offX, offY, wL, hL)));
        Mat leftMask = Mat::zeros(canvasH, canvasW, CV_8UC1);
        leftMask(Rect(offX, offY, wL, hL)).setTo(255);

        // ---------- (3a) 重叠区曝光增益补偿 ----------
        Mat overlapMask;
        bitwise_and(leftMask, warpedMask, overlapMask);
        const double g = estimateGain(canvas, warpedRight, overlapMask);
        cout << "估计增益 g = " << g
             << "（右图乘 g；理论约 1/0.88 ≈ 1.1364）" << endl;

        Mat leftF, rightF;
        canvas.convertTo(leftF, CV_32FC3);
        warpedRight.convertTo(rightF, CV_32FC3);
        rightF *= static_cast<float>(g);
        rightF = min(max(rightF, 0.0), 255.0);

        // ---------- (3b) 重叠区线性羽化 ----------
        int x0 = -1, x1 = -1;
        for (int x = 0; x < canvasW; ++x) {
            for (int y = 0; y < canvasH; ++y) {
                if (leftMask.at<uchar>(y, x) && warpedMask.at<uchar>(y, x)) {
                    if (x0 < 0) x0 = x;
                    x1 = x;
                    break;
                }
            }
        }
        if (x0 < 0) {
            throw runtime_error("未找到重叠区域");
        }
        const int ovW = x1 - x0 + 1;
        const int B = FEATHER_BANDWIDTH <= 0 ? ovW : min(FEATHER_BANDWIDTH, ovW);
        cout << "画布: " << canvasW << " x " << canvasH
             << ", 左图偏移: (" << offX << ", " << offY << ")"
             << ", 重叠宽度: " << ovW << " px, 羽化带宽: " << B << " px" << endl;
        if (ovW >= 0.9 * canvasW) {
            cerr << "警告: 两幅图像几乎完全重叠（H 接近单位阵、增益接近 1），"
                    "请检查 q8_left.png 与 q8_right.png 是否为同一张图，"
                    "或两张视图并未真正错开约 30%。" << endl;
        }

        // alpha：左图区域为 1，重叠区从 1 线性渐变到 0，右图独有区域为 0
        Mat alpha = Mat::zeros(canvasH, canvasW, CV_32FC1);
        alpha(Rect(offX, offY, wL, hL)).setTo(1.0f);
        for (int x = 0; x < ovW; ++x) {
            const float a = (ovW == 1) ? 0.5f
                                       : 1.0f - static_cast<float>(x) / static_cast<float>(ovW - 1);
            alpha.col(x0 + x).setTo(a);
        }

        Mat alphaInv = 1.0 - alpha;
        Mat a3, ai3;
        merge(vector<Mat>{alpha, alpha, alpha}, a3);
        merge(vector<Mat>{alphaInv, alphaInv, alphaInv}, ai3);

        Mat resultF = leftF.mul(a3) + rightF.mul(ai3);
        Mat result;
        resultF.convertTo(result, CV_8UC3);

        imwrite(outPath, result);
        cout << "已保存全景图: " << outPath << endl;

        // 直接输出结果：弹出窗口显示全景图（无图形环境时自动跳过）
        if (SHOW_RESULT) {
            try {
                namedWindow("Panorama", WINDOW_AUTOSIZE);
                imshow("Panorama", result);
                cout << "按任意键关闭结果窗口" << endl;
                waitKey(0);
                destroyAllWindows();
            } catch (const cv::Exception&) {
                cout << "当前环境无法弹窗显示，结果已保存到文件: " << outPath << endl;
            }
        }
    } catch (const exception& e) {
        cerr << "错误: " << e.what() << endl;
        return 1;
    }
    return 0;
}
