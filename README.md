# SphereSizeFinder

SphereSizeFinder is a Python-based tool that utilizes OpenCV library to accurately measure the diameters of spheres in digital images. With a straightforward graphical interface, it simplifies the process of identifying spheres, measuring their sizes, and classifying them based on diameter. This tool is particularly useful for the tasks that reqiures precise, contactless and high-throughput measurements.

## Features

- Accurate measurement of sphere diameters in images using the Hough Circle Transform from OpenCV.
- Customizable parameters for sphere detection to enhance accuracy and reduce false positives.
- Additional brightness parameter to exclude incorrectly identified spheres based on their brightness levels.
- K-means clustering for categorizing spheres into groups based on their sizes.
- Exporting functionality for both measurement data in CSV format and processed images in JPG format.
- A user-friendly graphical interface built with tkinter.

## Prerequisites

- Python 3.8 or higher
- OpenCV
- scikit-learn
- Pillow

## Installation

1. **Clone the Repository**

```
git clone https://github.com/HongxiBio/SphereSizeFinder.git
```

2. **Install Dependencies**

   Navigate to the SphereSizeFinder directory and install the required packages using pip:

```
cd SphereSizeFinder
pip install -r requirements.txt
```

4. **Run SphereSizeFinder**

Launch the tool by running:

```
python scripts/SphereSizeFinder-ENG.py
```

## Usage

After launching SphereSizeFinder, the graphical interface will guide you through the following steps:

1. **Import Image**: Load the image containing spheres you wish to measure.
2. **Set Parameters**: Adjust the detection parameters as needed to accurately identify spheres. Default values are provided, but adjustments may be necessary depending on your image.
3. **Measure and Classify**: The tool will automatically detect spheres, measure their diameters, and optionally classify them into groups using k-means clustering.
4. **Export Results**: Save the measurement data and/or processed image for further analysis or documentation.

**Important**:  
To accurately measure the diameter of spheres, the presence of a reference circle within the input image is essential. And the parameter of diameter should be changed too. This reference circle must have a markedly different size compared to the spheres being measured. It serves as a scale for calibrating measurements, ensuring the precision of diameter calculations.

## Contributing

Contributions to SphereSizeFinder are welcome! Whether it's adding new features, improving existing ones, or reporting bugs, any help is appreciated. Please feel free to fork the repository and submit pull requests.

## License

SphereSizeFinder is licensed under the MIT License. See the LICENSE file for more details.

---
