# SphereSizeFinder Code Description
This document provides a detailed overview of the code development.

## Overview
#### Purpose of the Program:
The program is designed to measure the diameters of spheres from images accurately and efficiently without physical contact.
#### Input:
Images: Contains spheres whose diameters are to be measured and a reference object (a known circle) for scale calibration.
#### Output:
- Annotated Images: Images marked with the detected spheres and reference circle.
- Average Diameter: Computed average diameter of all detected spheres.
- Diameter List: Individual diameters of each detected sphere.
- Grouped Diameter List: (Optional) Diameters grouped by similar sizes and their average diameters.

## Overview
#### Purpose of the Program:
To enable high-throughput, non-contact measurement of sphere diameters.

#### Input:
Images that include both the spheres to be measured and a reference circle for scale calibration.

#### Output:
- Images annotated with recognition results.
- Average diameter of the measured spheres.
- List of each sphere's diameter.
- Optionally, a grouped list of sphere diameters along with their average sizes.

---
## Algorithm Logic
#### Solution Approach:
1. Preprocess the image using the OpenCV library to enhance feature detection.
2. Use the **Hough Circle Transform** function from OpenCV to detect circles in the image:
   1. Initially, identify the reference circle by setting size-related parameters.
   2. Subsequently, detect spheres by adjusting parameters for roundness and diameter.
3. Exclude false positives by analyzing the average **grayscale value** within the bounds of the detected circles.
4. Results Output and Analysis:
   1. Mark the detected circles on the original image for visual verification.
   2. Record the diameters in a list.
   3. If required, use the clustering algorithms from `scikit-learn` to group diameters and compute their averages.
5. Develop a user interface (UI) to facilitate the use of the above functionalities.

# Code Implementation
### 1. Load Modules
```python
import cv2
import csv
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import tkinter.messagebox
```
1. `cv2`: This module from the OpenCV library is used for computer vision tasks such as image processing and feature extraction.
2. csv: A module from Python's standard library for handling CSV (Comma-Separated Values) files.
3. `numpy as np`: NumPy library, utilized for high-performance numerical computing, supports extensive operations on multi-dimensional arrays and matrices.
4. `from sklearn.cluster import KMeans`: Imports the KMeans clustering algorithm from the scikit-learn library.
5. `from PIL import Image, ImageTk`: Imports image processing and Tkinter image interface capabilities from the PIL (Python Imaging Library).
6. `import tkinter as tk` and `from tkinter import ttk`: These modules import the Tkinter library, used for creating graphical user interfaces (GUIs).
7. `import tkinter.filedialog` and `import tkinter.messagebox`: These modules from the Tkinter library provide file dialog and message box functionalities, used for file selection and user interaction

### 2. Defining Functions
#### `dir`:
```python
def dir():
    path = tk.filedialog.askopenfilename()
    filename.set(path)
```
This function lets the user select a file via a graphical interface and returns its path.

#### `cv_imread`:
```python
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img
```
Reads an image file with a specified path through the OpenCV library and returns the decoded image data.  
#### `image_edit`:
```python
def image_edit(img):
    image = cv2.resize(img, (3200, int(img.shape[0] * 3200 / img.shape[1])))
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
    return (image,blurred_image)
```
The function `image_edit` takes an image `img` as input and returns the resized original color image and the blurred grayscale image.  

#### `get_ref_param`:
```python
def get_ref_param():
    ref_param = {}
    ref_param["mindistance"] = float(E1.get())
    ref_param["param1"] = float(E2.get())
    ref_param["param2"] = float(E3.get())
    ref_param["minRadius"] = int(E4.get())
    ref_param["maxRadius"] = int(E5.get())
    return ref_param
```
Collects the parameters of the reference circles entered by the user through the graphical interface and organizes these parameters into a dictionary form.  
`mindistance`:Distance limit from the center of the recognized circles, used to exclude circles with overlapping.  
`param1`,`param2`:Edge sharpness and roundness of the ball.  
`minRadius`,`maxRadius`: size limit of the ball.  

#### `get_target_param`:
```python
    target_param = {}
    target_param["mindistance"] = float(E6.get())
    target_param["param1"] = float(E7.get())
    target_param["param2"] = float(E8.get())
    target_param["minRadius"] = int(E9.get())
    target_param["maxRadius"] = int(E10.get())
    target_param["minBright"] = int(E12.get())
    return target_param
```
Collects the parameters of the ball to be measured, entered by the user through the graphical interface, and organizes these parameters into a dictionary form.  
`mindistance`: limit on the distance from the center of the recognized circles, used to exclude circles with overlap.  
`param1`,`param2`:Edge sharpness and roundness of the ball.  
`minRadius`,`maxRadius`: size limit of the ball.  
`minBright`: the minimum brightness of the circle, used to exclude fake circles.  

#### `calculate_average_variance`:
```python
def calculate_average_variance(numbers):  
    if len(numbers) == 0:  
        return 0.0, 0.0  
        
    average = sum(numbers) / len(numbers)  
    variance = sum((x - average) ** 2 for x in numbers) / len(numbers) 
    average = "{:.2f}".format(average)
    variance = "{:.2f}".format(variance) 

    return (average, variance)
```
This function calculates the mean and variance of a list of input numbers and returns both values in two decimal format.   

#### `calc_circle_grey`:
```python
def calc_circle_grey(circle_info,img):
    x, y, r= circle_info
    center_coordinates = (int(x), int(y))  
    radius = int(r)

    x_indices, y_indices = np.ogrid[-radius:radius+1, -radius:radius+1]  
    mask = x_indices**2 + y_indices**2 <= radius**2

    circle_pixels = img[center_coordinates[1]-radius:center_coordinates[1]+radius+1,     
                        center_coordinates[0]-radius:center_coordinates[0]+radius+1]  
    circle_pixels = circle_pixels[mask]
    average_gray_value = np.mean(circle_pixels)

    return(average_gray_value)
```
Based on the given circle center and radius, pixels within the circular region are extracted from the input image `img` and the average gray value of these pixels is calculated. The average gray value will be returned, as an indicator of the brightness of the sphere to be measured.  

#### `cluster_balls`:
```python
def cluster_balls(sizes, n_clusters):

    sizes = np.array(sizes).reshape(-1, 1)
    
    # Clustering with KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(sizes)
    
    results = {}
    
    for i in range(n_clusters):
        cluster_sizes = sizes[kmeans.labels_ == i].ravel()
        
        mean = np.mean(cluster_sizes)
        variance = np.var(cluster_sizes)
        
        # Save the values
        results[f'Group {i+1}'] = {
            'Sizes': cluster_sizes,
            'Mean': mean,
            'Variance': variance
        }
    return results
```
The function `cluster_balls` receives as input a list of ball sizes `sizes` and the number of clusters to be formed `n_clusters`. The ball sizes are clustered using the KMeans algorithm and the mean size and variance of each cluster is calculated. Finally, a dictionary is returned containing the ball sizes, mean and variance in each cluster.  

#### `save_diameter`:
```python
def save_diameter(diameter_dic):

    for key in diameter_dic:
        if not isinstance(diameter_dic[key], list):
            diameter_dic[key] = [diameter_dic[key]]
    data_rows = [[key] + value if isinstance(value, list) else [key, value] for key, value in diameter_dic.items()]  

    out_path = tk.filedialog.asksaveasfilename(defaultextension=".csv")
    if out_path:
        with open(out_path,'w',newline='',encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["No.","D(mm)"])
            writer.writerows(data_rows)
        tkinter.messagebox.showinfo(title="message", message=f"Sphere diameter data has been successfully saved to {out_path}")
```
Function for saving data, which converts the received diameter data dictionary into a two-dimensional list in CSV format and allows the user to save the data as a CSV file by popping up a dialog box and notifying the user of the successful saving.  

#### `save_image`：
```python
def save_image(image):  
    img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
 
    out_path = tk.filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg;*.jpeg"), ("PNG", "*.png"), ("All Files", "*.*")])  
    if out_path:  
 
        img.save(out_path)  
  
        tk.messagebox.showinfo("message", f"The recognition image has been successfully saved to {out_path}") 
```
Function for saving image, which converts the received image from BGR to RGB format, then saves it to the file path specified by the user via and pops up a message alert box to notify the user upon success.  

#### `ident_circles`:
```python
def ident_circles(img0, blurimg, ref_param, target_param, refdiamter):
    ref_circle = cv2.HoughCircles(blurimg, cv2.HOUGH_GRADIENT, 0.5, 
                                  ref_param["mindistance"], 
                                  param1=ref_param["param1"], 
                                  param2=ref_param["param2"], 
                                  minRadius=ref_param["minRadius"], 
                                  maxRadius=ref_param["maxRadius"])
    if ref_circle is not None:
        ref_circle = np.uint16(np.around(ref_circle))
        diameters = {}

        for i in ref_circle[0, :]:
            cv2.circle(img0, (i[0], i[1]), i[2], (255, 0, 0), 8)
            cv2.circle(img0, (i[0], i[1]), 2, (0, 0, 255), 5) 
        
        if len(ref_circle[0]) == 1: 
            pixel_diamter = 2*ref_circle[0][0][2]  
            pixel_unit = refdiamter/pixel_diamter 
            

            circles = cv2.HoughCircles(blurimg, cv2.HOUGH_GRADIENT, 0.5,
                                       target_param["mindistance"], 
                                       param1=target_param["param1"], 
                                       param2=target_param["param2"], 
                                       minRadius=target_param["minRadius"], 
                                       maxRadius=target_param["maxRadius"])
            if circles is not None:
                circles = np.uint16(np.around(circles))
                j = 0
                for i in circles[0, :]:
                    # calculate  the average grayscale value of a sphere
                    grey_value = calc_circle_grey(i,img0)
                    if grey_value >= target_param["minBright"]:
                        # draw the sphere to the image
                        cv2.circle(img0, (i[0], i[1]), i[2], (255, 0, 255), 2)
                        cv2.circle(img0, (i[0], i[1]), 2, (0, 255, 0), 2)
                        # label the sphere
                        j += 1
                        cv2.putText(img0, str(j), (i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 174, 0), 2)

                        # calculate  the diameter
                        diameters[j] = i[2] * 2 * pixel_unit
            else:
                tkinter.messagebox.showinfo(title="Missing target", message="No spheres were detected. Try to adjust the parameters of Hough circle transformation.")
        else:
            tkinter.messagebox.showinfo(title="Unexpected references",message=f"{len(ref_circle[0])}reference circles detected, please review and adjust the Hough circle transformation parameters")
        
        return (img0,diameters)
    else:
        tkinter.messagebox.showinfo(title="Missing reference", message="No reference circle was detected. Try to adjust the parameters of Hough circle transformation.")
```
Core functions for the recognition of the ball to be measured:
1. Recognizes the presence of a reference circle and obtains the corresponding length of each pixel;
2. Identify the balls to be measured and filter the valid balls by their gray value.
3. Record the diameter of the ball to be measured and label it in the original picture.
4. It return:
   1. an image with the detected circles plotted
   2. a dictionary containing the diameters of the circles

```python
def display(results):  
    diameters = results[1]
    img0 = results[0]
    img0 = cv2.resize(img0,(800,int(800*img0.shape[0]/img0.shape[1])), interpolation=cv2.INTER_AREA)
    
    img = Image.fromarray(cv2.cvtColor(img0,cv2.COLOR_BGR2RGB))
    tk_img = ImageTk.PhotoImage(image=img)
    
    ave, var = calculate_average_variance(list(diameters.values())) 

    result = tk.Toplevel() 
    result.title("Results")
    label_img = tk.Label(result, image=tk_img)
    label_img.grid(row=1, column=1,columnspan=2)

    label_summary = tk.Label(result, text=f"average diameter:{ave}; variance：{var}")
    label_summary.grid(row=2,column=2)
    
    tree = ttk.Treeview(result,show="headings")
    tree["columns"] = ("NO","diameters")
    tree.column("NO", width=40, minwidth=40, stretch=tk.NO)
    tree.column("diameters", width=100, minwidth=100, stretch=tk.NO)
    tree.heading("NO", text="NO.", anchor=tk.W)
    tree.heading("diameters", text="diameter(mm)", anchor=tk.W)
    for key, value in diameters.items():
        tree.insert("", tk.END, values=(key,f"{value:.3f}"))
    tree.grid(row=3, column=1, rowspan=3)

    label_tree = tk.Label(result,text="diameter list")
    label_tree.grid(row=2, column=1)
    

    group_number_label = tk.Label(result, text="No. of groups needed to be separated")
    group_number_label.grid(row=3,column=2)
    groups_number = tk.Entry(result, width=5)
    groups_number.grid(row=4,column=2)
    groups_number.insert(tk.END,"2")
    
    n_of_clusters = int(groups_number.get())
    analy_data = [list(diameters.values()),n_of_clusters]

    analy_button = tk.Button(result, text="cluster analysis",command=lambda:analy_window(analy_data))
    analy_button.grid(row=6,column=2)
    save_button = tk.Button(result, text="Export diameter information", command=lambda:save_diameter(diameters))
    save_button.grid(row=7,column=1,pady=20)
    save_button_img = tk.Button(result, text="Export recognition images", command=lambda:save_image(results[0]))
    save_button_img.grid(row=7,column=2,pady=20)

    result.mainloop()
```
This function is used to display the recognition results in a separate window containing:
- An image
- Ball diameter data to be measured
- Clustering interaction options
- Saving options

#### `analy_window`:
```python
def analy_window(class_info):
    analy_result = cluster_balls(class_info[0], class_info[1])
    
    anal_window = tk.Toplevel()
    anal_window.title("Clustering results")

    anal_window.geometry("400x400")

    label_info = tk.Label(anal_window, text=f"Classified into {len(analy_result)} groups")
    label_info.grid(row=0, column=0, columnspan=len(analy_result)) 
    
    column = 0
    for group, info in analy_result.items():
        frame = tk.Frame(anal_window)
        frame.grid(row=1, column=column, padx=10, pady=10, sticky="nw")

        label = tk.Label(frame, text=f"{group}\n detected{len(info['Sizes'])}spheres\n average diameter: {info['Mean']:.3f}mm\n variance: {info['Variance']:.3f}")
        label.grid()

        value_tree = ttk.Treeview(frame, show="headings", columns=("diameters"), height=5)
        value_tree.column("diameters", width=100, minwidth=100, stretch=tk.NO)
        value_tree.heading("diameters", text="diameter(mm)", anchor=tk.W)
        for value in info['Sizes']:
            value_tree.insert("", tk.END, values=(f"{value:.3f}",)) 
        value_tree.grid(rowspan=2)

        column += 1 

    anal_window.mainloop()
```
The main purpose of this function is to display the results of the cluster analysis. It calls the `cluster_balls` function to perform a cluster analysis on the input `class_info` and then displays the results in a newly opened top-level window.

### Defining main logic
#### `main`:
```python
def main():
    ref_param = get_ref_param()
    target_param = get_target_param()
    img0 = cv_imread(filename.get())
    img, blurimg = image_edit(img0)
    refdiamter = float(E11.get())
    results = ident_circles(img0=img,blurimg=blurimg,ref_param=ref_param,target_param=target_param,refdiamter=refdiamter)
    display(results=results)
```
This function is the main startup function of the recognition program and specifies the order in which the functions are executed.

### GUI production
```python
window=tk.Tk()
window.title("SphereSizeFinder")
window.geometry("300x400")

filename = tk.StringVar()

tk.Button(window, text = "Import Images", command = dir).grid(row=1, column=0, padx=5, pady=5)

L1 = tk.Label(window, text="Minimum distance between different reference circles' center").grid(row=2, column=0)
E1 = tk.Entry(window, width=5)
E1.grid(row=3, column=0)
E1.insert(tk.END, "200") 

L2 = tk.Label(window, text="Edge sharpness of reference circle").grid(row=5, column=0)
E2 = tk.Entry(window, width=5)
E2.grid(row=6, column=0)
E2.insert(tk.END, "100") 

L3 = tk.Label(window, text="Roundness of reference circle").grid(row=8, column=0)
E3 = tk.Entry(window, width=5)
E3.grid(row=9, column=0)
E3.insert(tk.END, "80") 

L4 = tk.Label(window, text="Minimum radius of reference circle").grid(row=11, column=0)
E4 = tk.Entry(window, width=5)
E4.grid(row=12, column=0)
E4.insert(tk.END, "200") 

L5 = tk.Label(window, text="Maximum radius of reference circle").grid(row=14, column=0)
E5 = tk.Entry(window, width=5)
E5.grid(row=15, column=0)
E5.insert(tk.END, "350") 

L11 = tk.Label(window,text="Diameter of reference circle(mm)").grid(row=17,column=0)
E11 = tk.Entry(window, width=5)
E11.grid(row=18,column=0)
E11.insert(tk.END, "25.00") 

L6 = tk.Label(window, text="Minimum distance between different spheres").grid(row=2, column=20)
E6 = tk.Entry(window, width=5)
E6.grid(row=3, column=20)
E6.insert(tk.END, "15") 

L7 = tk.Label(window, text="Edge sharpness of spheres").grid(row=5, column=20)
E7 = tk.Entry(window, width=5)
E7.grid(row=6, column=20)
E7.insert(tk.END, "100") 

L8 = tk.Label(window, text="Roundness of the spheres").grid(row=8, column=20)
E8 = tk.Entry(window, width=5)
E8.grid(row=9, column=20)
E8.insert(tk.END, "15") 

L9 = tk.Label(window, text="Minimum radius of spheres").grid(row=11, column=20)
E9 = tk.Entry(window, width=5)
E9.grid(row=12, column=20)
E9.insert(tk.END, "20") 

L10 = tk.Label(window, text="Maximum radius of spheres").grid(row=14, column=20)
E10 = tk.Entry(window, width=5)
E10.grid(row=15, column=20)
E10.insert(tk.END, "50") 

L12 = tk.Label(window, text="Minimum brightness of spheres (0-255)").grid(row=17, column=20)
E12 = tk.Entry(window, width=5)
E12.grid(row=18, column=20)
E12.insert(tk.END, "0") 

submit_button = tk.Button(window, text="Search", command=main).grid(row=22,column=10)

window.mainloop()
```
Create the UI of the main interface with the tinkter library and assign default values to each parameter. **Following default values need to be changed can be adjusted here directly parameter values **.