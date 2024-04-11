# SphereSizeFinder 球体尺寸测量器

SphereSizeFinder 是一个基于 Python 的工具，利用opencv库准确测量数字图像中球体的直径。通过直观的图形界面，简化了识别球体、测量其尺寸并根据直径分类的过程。此工具特别适用于需要精确、非接触式、高通量测量的任务。

## 功能

- 使用 OpenCV 的霍夫圆变换算法识别并标记图像中的球体。
- 以CSV格式导出识别出的球体直径。
- 自定义的球体检测参数，用以提高识别准确度。
- 可选的的亮度阈值参数，进一步降低误判。
- 支持以CSV格式导出测量数据、以及以JPG格式导出处理后的图像。
- （可选）使用 K-means 聚类算法，将球体依据直径大小分类。

## 所需依赖

- Python 3.8 或更高版本
- OpenCV
- scikit-learn
- Pillow

## 安装

1. **克隆仓库**

 ```
 git clone https://github.com/HongxiBio/SphereSizeFinder.git
 ```

2. **安装依赖**

导航至 SphereSizeFinder 目录，并使用 pip 安装所需的包：

 ```
cd SphereSizeFinder
pip install -r requirements.txt
 ```

4. **运行 SphereSizeFinder**

 通过运行以下命令启动工具：

 ```
 python SphereSizeFinder.py
 ```

## 使用说明

启动 SphereSizeFinder 后，根据图形界面完成以下步骤：

1. **导入图片**：加载含有您希望测量的球体的图片。
2. **设置参数**：根据需要调整检测参数以准确识别球体。提供默认值，但根据您的图片可能需要调整。
3. **测量与分类**：工具将自动检测球体，测量其直径并展示出来。
4. **聚类（可选）**：如果途中有多种不同大小的球，您还可以选择性地使用 k-means 聚类算法将它们分类。
5. **导出结果**：可以选择保存测量数据以及处理后的图片供后续分析。

**重要提示**：  
为了准确测量球体直径，输入图像中必须包含一个参考圆。该参考圆的尺寸必须与被测量的球体明显不同。它用作校准测量的标尺，确保直径计算的精确性。

## 贡献

欢迎对 SphereSizeFinder 进行贡献！无论是添加新功能、改进现有功能还是报告错误，任何建议与帮助都是受到欢迎的。请随意分叉仓库并提交拉取请求。

## 许可证

SphereSizeFinder 根据 MIT 许可证授权。有关更多细节，请查看 LICENSE 文件。