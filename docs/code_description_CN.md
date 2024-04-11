# 球体尺寸测量器代码说明
本说明拆解详述了代码作用以及潜在缺陷，用以后续开发。
## 概览
#### 程序目的：
实现球体直径的高通量、非接触式测量。
#### 输入：  
同时含有待测球和参照圆的图片  
  
#### 输出：  
- 含有识别结果的图片
- 待测球的平均直径
- 待测球大小列表
- 分组的待测球大小列表与均值（可选）

---
## 算法逻辑
#### 解决思路：
1. 使用opencv库预处理图片
2. 使用opencv库的**霍夫圆变换**函数识别图内存在的圆
   1. 首先通过设置圆的参数（主要是大小）来识别参考圆
   2. 再识别待测球，通过调整圆润度、大小等参数识别待测球
3. 通过判断识别结果的圆内的平均**灰度值**来排除假识别结果
4. 结果输出与分析：  
   1. 在原图中标注
   2. 输出直径到列表
   3. 如有分组需求，使用sk-learn算法对直径进行聚类。
5. 为以上功能设计UI

# 代码实现
### 1. 加载模块
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
1. `cv2`：OpenCV库，用于计算机视觉任务，如图像处理、特征提取等。
2. `csv`：Python标准库中的CSV模块，用于处理CSV（逗号分隔值）文件。
3. `numpy as np`：NumPy库，用于进行高效的数值计算，支持大量的维度数组与矩阵运算。
4. `from sklearn.cluster import KMeans`：从scikit-learn库中导入KMeans聚类算法。
5. `from PIL import Image, ImageTk`：从PIL（Python Imaging Library）库中导入图像处理和Tkinter图像接口功能。
6. `import tkinter as tk` 和 `from tkinter import ttk`：导入Tkinter库，用于创建图形用户界面（GUI）。
7. `import tkinter.filedialog` 和 `import tkinter.messagebox`：从Tkinter库中导入文件对话框和消息框功能，用于文件选择和用户交互。
### 2. 定义函数
#### `dir`:
```python
def dir():
    path = tk.filedialog.askopenfilename()
    filename.set(path)
```
这个函数通过图形界面让用户选择一个文件，并返回其路径

#### `cv_imread`:
```python
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img
```
通过OpenCV库读取指定路径的图像文件，并返回解码后的图像数据。
#### `image_edit`:
```python
def image_edit(img):
    image = cv2.resize(img, (3200, int(img.shape[0] * 3200 / img.shape[1])))
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
    return (image,blurred_image)
```
该函数`image_edit`接收一个图像`img`作为输入,返回调整大小后的原始彩色图像和经过模糊处理的灰度图像。
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
收集用户通过图形界面输入的参考圆的参数，并将这些参数组织成一个字典形式。
`mindistance`:识别出来的圆的圆心距离限制，用于排除有重叠的圆。
`param1`,`param2`:球的边缘清晰度和圆润度。
`minRadius`,`maxRadius`:球的大小限制
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
收集用户通过图形界面输入的待测球的参数，并将这些参数组织成一个字典形式。  
`mindistance`:识别出来的圆的圆心距离限制，用于排除有重叠的圆。  
`param1`,`param2`:球的边缘清晰度和圆润度。  
`minRadius`,`maxRadius`:球的大小限制。  
`minBright`：圆的最小亮度，用于排除假圆。  

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
该函数计算输入数字列表的平均值和方差，并以两位小数的格式返回这两个值。

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
根据给定的圆心和半径信息，从输入图像`img`中提取圆形区域内的像素，并计算这些像素的平均灰度值。最后，返回该平均灰度值,作为待测球亮度指标。

#### `cluster_balls`:
```python
def cluster_balls(sizes, n_clusters):

    sizes = np.array(sizes).reshape(-1, 1)
    
    # 使用KMeans进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(sizes)
    
    results = {}
    
    for i in range(n_clusters):
        # 获取当前聚类的成员
        cluster_sizes = sizes[kmeans.labels_ == i].ravel()
        
        # 计算平均值和方差
        mean = np.mean(cluster_sizes)
        variance = np.var(cluster_sizes)
        
        # 保存结果
        results[f'Group {i+1}'] = {
            'Sizes': cluster_sizes,
            'Mean': mean,
            'Variance': variance
        }
    return results
```
该函数`cluster_balls`接收球的大小列表`sizes`和要形成的聚类数量`n_clusters`作为输入。使用KMeans算法对球的大小进行聚类，并计算每个聚类的平均大小和方差。最后，返回一个字典，其中包含每个聚类中的球大小、平均值和方差。
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
        tkinter.messagebox.showinfo(title="信息", message=f"待测球直径数据已成功保存到{out_path}")
```
保存函数，该函数将接收的直径数据字典转换为CSV格式的二维列表，并通过弹出对话框让用户将数据保存为CSV文件，并通知用户保存成功。
#### `save_image`：
```python
def save_image(image):  
    img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    # 显示保存文件对话框  
    out_path = tk.filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg;*.jpeg"), ("PNG", "*.png"), ("All Files", "*.*")])  
    if out_path:  
        # 保存图像到指定路径  
        img.save(out_path)  
        # 显示成功信息  
        tk.messagebox.showinfo("信息", f"识别图片已成功保存到 {out_path}") 
```
保存函数，该函数将接收的图像从BGR转换为RGB格式，然后保存为用户通过指定的文件路径，并在成功后弹出信息提示框通知用户。
#### `display`:
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
            cv2.circle(img0, (i[0], i[1]), 2, (0, 0, 255), 5)  # 绘制圆心和圆轮廓
        
        if len(ref_circle[0]) == 1: 
            pixel_diamter = 2*ref_circle[0][0][2]  
            pixel_unit = refdiamter/pixel_diamter                  # 计算单位像素的长度
            
            # 检测圆形
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
                    # 检测圆的平均内灰度值
                    grey_value = calc_circle_grey(i,img0)
                    if grey_value >= target_param["minBright"]:
                        # 绘制圆心和圆轮廓
                        cv2.circle(img0, (i[0], i[1]), i[2], (255, 0, 255), 2)
                        cv2.circle(img0, (i[0], i[1]), 2, (0, 255, 0), 2)
                        # 标注圆
                        j += 1
                        cv2.putText(img0, str(j), (i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 174, 0), 2)

                        # 计算圆的直径
                        diameters[j] = i[2] * 2 * pixel_unit
            else:
                tkinter.messagebox.showinfo(title="待测球缺失", message="未检测到成型的待测球，如若需要，可调整霍夫圆变换参数")
        else:
            tkinter.messagebox.showinfo(title="额外的参照圆",message=f"检测到,{len(ref_circle[0])}个参照圆，请查看并调整霍夫圆变换参数")
        
        return (img0,diameters)
    else:
        tkinter.messagebox.showinfo(title="参照圆缺失", message="未检测到参照圆，请调整参照圆参数")
```
待测球识别的核心函数：
1. 识别参考圆的存在并获取每个像素对应的长度；
2. 识别待测球，并通过灰度值来筛选有效的待测球
3. 记录待测球直径信息并标注在原图中
4. 返回：
   1. 绘制了检测到的圆的图像
   2. 一个包含圆的直径的字典

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

    label_summary = tk.Label(result, text=f"平均直径为:{ave}; 方差：{var}")
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

    label_tree = tk.Label(result,text="直径数据")
    label_tree.grid(row=2, column=1)
    

    group_number_label = tk.Label(result, text="需要分开的组数")
    group_number_label.grid(row=3,column=2)
    groups_number = tk.Entry(result, width=5)
    groups_number.grid(row=4,column=2)
    groups_number.insert(tk.END,"2")
    
    n_of_clusters = int(groups_number.get())
    analy_data = [list(diameters.values()),n_of_clusters]

    analy_button = tk.Button(result, text="聚类分析",command=lambda:analy_window(analy_data))
    analy_button.grid(row=6,column=2)
    save_button = tk.Button(result, text="导出直径信息", command=lambda:save_diameter(diameters))
    save_button.grid(row=7,column=1,pady=20)
    save_button_img = tk.Button(result, text="导出识别图片", command=lambda:save_image(results[0]))
    save_button_img.grid(row=7,column=2,pady=20)

    result.mainloop()

```
函数用于在一个单独的窗口中显示识别结果，其中包括：
- 一张图像
- 待测球直径数据
- 聚类交互选项
- 保存选项
#### `analy_window`:
```python
def analy_window(class_info):
    analy_result = cluster_balls(class_info[0], class_info[1])
    
    anal_window = tk.Toplevel()
    anal_window.title("聚类结果")
    # 根据Treeview数量和内容调整窗口大小
    anal_window.geometry("400x400")

    label_info = tk.Label(anal_window, text=f"总共分为了{len(analy_result)}组")
    label_info.grid(row=0, column=0, columnspan=len(analy_result))  # 调整为跨越所有列
    
    column = 0
    for group, info in analy_result.items():
        frame = tk.Frame(anal_window)
        frame.grid(row=1, column=column, padx=10, pady=10, sticky="nw")

        label = tk.Label(frame, text=f"{group}\n总共{len(info['Sizes'])}个球\n均值: {info['Mean']:.3f}mm\n方差: {info['Variance']:.3f}")
        label.grid()

        value_tree = ttk.Treeview(frame, show="headings", columns=("diameters"), height=5)
        value_tree.column("diameters", width=100, minwidth=100, stretch=tk.NO)
        value_tree.heading("diameters", text="diameter(mm)", anchor=tk.W)
        for value in info['Sizes']:
            value_tree.insert("", tk.END, values=(f"{value:.3f}",))  # 注意values是元组形式
        value_tree.grid(rowspan=2)

        column += 1  # 移动到下一列以并排显示

    anal_window.mainloop()
```
这个函数主要作用为显示聚类分析的结果。它调用 `cluster_balls` 函数对输入的 `class_info` 进行聚类分析，然后将结果显示在一个新打开的顶层窗口中。

### 定义运行逻辑
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
这个函数是识别程序的主要启动函数，规定了函数执行的顺序。

### 主界面UI制作
```python
window=tk.Tk()
window.title("待测球识别器")
window.geometry("300x400")

filename = tk.StringVar()

tk.Button(window, text = "导入图片", command = dir).grid(row=1, column=0, padx=5, pady=5)

L1 = tk.Label(window, text="参照圆圆心最小距离").grid(row=2, column=0)
E1 = tk.Entry(window, width=5)
E1.grid(row=3, column=0)
E1.insert(tk.END, "200") 

L2 = tk.Label(window, text="参照圆边缘清晰度").grid(row=5, column=0)
E2 = tk.Entry(window, width=5)
E2.grid(row=6, column=0)
E2.insert(tk.END, "100") 

L3 = tk.Label(window, text="参照圆圆润度").grid(row=8, column=0)
E3 = tk.Entry(window, width=5)
E3.grid(row=9, column=0)
E3.insert(tk.END, "80") 

L4 = tk.Label(window, text="参照圆最小半径").grid(row=11, column=0)
E4 = tk.Entry(window, width=5)
E4.grid(row=12, column=0)
E4.insert(tk.END, "200") 

L5 = tk.Label(window, text="参照圆最大半径").grid(row=14, column=0)
E5 = tk.Entry(window, width=5)
E5.grid(row=15, column=0)
E5.insert(tk.END, "350") 

L11 = tk.Label(window,text="参照圆直径(mm)").grid(row=17,column=0)
E11 = tk.Entry(window, width=5)
E11.grid(row=18,column=0)
E11.insert(tk.END, "25.00") 

L6 = tk.Label(window, text="待测球相互距离(半径))").grid(row=2, column=20)
E6 = tk.Entry(window, width=5)
E6.grid(row=3, column=20)
E6.insert(tk.END, "15") 

L7 = tk.Label(window, text="待测球边缘清晰度").grid(row=5, column=20)
E7 = tk.Entry(window, width=5)
E7.grid(row=6, column=20)
E7.insert(tk.END, "100") 

L8 = tk.Label(window, text="待测球圆润度").grid(row=8, column=20)
E8 = tk.Entry(window, width=5)
E8.grid(row=9, column=20)
E8.insert(tk.END, "15") 

L9 = tk.Label(window, text="待测球最小半径").grid(row=11, column=20)
E9 = tk.Entry(window, width=5)
E9.grid(row=12, column=20)
E9.insert(tk.END, "20") 

L10 = tk.Label(window, text="待测球最大半径").grid(row=14, column=20)
E10 = tk.Entry(window, width=5)
E10.grid(row=15, column=20)
E10.insert(tk.END, "50") 

L12 = tk.Label(window, text="待测球最小亮度0-255").grid(row=17, column=20)
E12 = tk.Entry(window, width=5)
E12.grid(row=18, column=20)
E12.insert(tk.END, "180") 

submit_button = tk.Button(window, text="查找", command=main).grid(row=22,column=10)

window.mainloop()
```
通过tinkter库制作主界面的UI并赋予各个参数默认值。**后续默认值需修改可在此直接调整参数值**。