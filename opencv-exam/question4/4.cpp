#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

// ---------- 可调参数 ----------
const float NL_H   = 5.0f;   // 输入去噪强度：默认 5（轻），过大才会糊字
const float BG_MIN = 10.0f;  // 背景下限：防除零、防暗区噪声爆炸
const int   SMOOTH = 3;      // 背景平滑核（只作用于非文字区，可改 5）

// 根据图像尺寸自适应生成奇数核大小
static int oddKernelSize(int rows, int cols, int divisor = 15) {
    int k = max(31, min(rows, cols) / divisor);
    if (k % 2 == 0) ++k;   // 中值滤波核必须是奇数
    return k;
}

int main(int argc, char** argv) {
    string path = (argc > 1) ? argv[1] : "../question4/picture4/image4.png";

    Mat src = imread(path, IMREAD_COLOR);
    if (src.empty()) {
        cerr << "无法读取图像: " << path << endl;
        return -1;
    }

    // ---------- 0. 原图保持颜色不变：另存备份，之后绝不修改 src ----------
    imwrite("original_color.png", src);

    // 转到 Lab：只处理亮度通道 L，颜色通道 a/b 原样保留 → 不产生色偏
    Mat lab;
    cvtColor(src, lab, COLOR_BGR2Lab);
    vector<Mat> ch(3);
    split(lab, ch);
    Mat L = ch[0];

    // ---------- 1. 输入轻去噪（保边，强度低，不糊字） ----------
    Mat Lc;
    fastNlMeansDenoising(L, Lc, NL_H, 7, 21);
    // 备选：bilateralFilter(L, Lc, 5, 30, 5);

    // ---------- 2. 估计背景光照场（在去噪后的 L 上估计更稳） ----------
    int k = oddKernelSize(Lc.rows, Lc.cols);   // 本图约 45×45
    Mat bg;
    medianBlur(Lc, bg, k);                     // 方法A：大尺度中值滤波（默认）
    // 方法B：形态学闭运算（文字较密时更稳），取消注释并注释掉上面两行
    // Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(k, k));
    // morphologyEx(Lc, bg, MORPH_CLOSE, kernel);

    // ---------- 3. 光照归一化：norm = L / bg ----------
    Mat fL, fbg;
    Lc.convertTo(fL, CV_32F);
    bg.convertTo(fbg, CV_32F);
    fbg.setTo(Scalar(BG_MIN), fbg < BG_MIN);   // 下限截断
    Mat fnorm;
    divide(fL, fbg, fnorm, 255.0);             // fnorm = fL * 255 / fbg
    Mat norm;
    fnorm.convertTo(norm, CV_8U);              // 自动截断到 [0, 255]

    // ---------- 4. 文字感知的背景降噪（关键：只平滑背景，不碰文字） ----------
    // 4a. Otsu 找出文字像素（暗像素 → 255）
    Mat textMask;
    threshold(norm, textMask, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    // 4b. 文字区轻微膨胀 1px，保护笔画边缘不被平滑
    Mat k3 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    dilate(textMask, textMask, k3);
    // 4c. 平滑整图，但只把"非文字像素"替换为平滑结果
    Mat bgSmooth;
    medianBlur(norm, bgSmooth, SMOOTH);
    Mat denoised = norm.clone();
    bgSmooth.copyTo(denoised, ~textMask);      // 文字像素保持原值 → 字迹清晰

    // ---------- 5. 合并回 Lab 并转回 BGR：颜色保持不变 ----------
    ch[0] = denoised;
    Mat labOut;
    merge(ch, labOut);
    Mat colorOut;
    cvtColor(labOut, colorOut, COLOR_Lab2BGR);
    imwrite("result_color.png", colorOut);

    // ---------- 6. 灰度 + 二值化，验证阴影/污渍区文字是否恢复 ----------
    Mat gray = denoised.clone();
    Mat enhanced;
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));  // 自适应对比度增强
    clahe->apply(gray, enhanced);

    Mat bin;
    threshold(enhanced, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
    // 若二值图背景仍有少量散点，再取消注释下面的 3×3 开运算
    // （会轻微腐蚀笔画，默认不开，保证字迹完整）
    // morphologyEx(bin, bin, MORPH_OPEN, k3);

    imwrite("result_gray.png", gray);
    imwrite("result_binary.png", bin);

    imshow("原图(颜色不变)", src);
    imshow("估计背景", bg);
    imshow("归一化+去噪", denoised);
    imshow("二值化验证", bin);
    cout << "完成：original_color / result_color / result_gray / result_binary" << endl;

    waitKey(0);
    return 0;
}
