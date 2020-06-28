// https://blog.csdn.net/zhhp1001/article/details/84816328
/**
 * FileStorage类将各种OpenCV数据结构的数据存储为XML 或 YML格式。
构造函数
cv::FileStorage(const string& source, int flags， const string& encoding=string());  

source –存储或读取数据的文件名（字符串），其扩展名(.xml 或 .yml或者.yaml)决定文件格式。
flags – 操作方式，包括：
    FileStorage::READ 打开文件进行读操作
    FileStorage::WRITE 打开文件进行写操作
    FileStorage::APPEND打开文件进行附加操作，在已有内容的文件里添加
encoding—编码方式，用默认值就好。 
 */


/**
 * FileStorage类的使用流程如下：
（1）实例化一个FileStorage类对象
（2）使用流操作符<<进行文件写入，>>进行文件读取，类似C++中的文件操作
（3）使用FileStorage::release()函数析构掉类对象，并关闭文件
 */


#include <opencv2/opencv.hpp>
// !下面是写操作的例子：

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
 
using namespace std;
using namespace cv;
 
int main(int argc, char** argv)
{
    FileStorage file_("1.yml", FileStorage::WRITE);
    int num1 = 1, num2 = 2;
    file_<<"num1"<<num1;
    file_<<"num2"<<num2;
    Mat A(4,5,CV_32FC1,Scalar(10));
    file_<<"A"<<A;
    file_.release();
}

// 输出1.yml文件内容为：
// %YAML:1.0
// ---
// num1: 1
// num2: 2
// A: !!opencv-matrix
//    rows: 4
//    cols: 5
//    dt: f
//    data: [ 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
//        10., 10., 10., 10., 10., 10., 10., 10. ]



// !下面是读操作的例子：

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
 
using namespace std;
using namespace cv;
 
int main(int argc, char** argv)
{
    FileStorage fs_ ("1.yml", FileStorage::READ);
    int r1, r2;
    fs_["num1"]>>r1;
    fs_["num2"]>>r2;   
    Mat A(4,5,CV_32FC1);
    fs_["A"]>>A;
    cout<< "r1 :"<<r1<<endl;
    cout<< "r2 :"<<r2<<endl;
    cout<< "A :"<<"\n"<<A<<endl;
    fs_.release();
}


// ! 特别注意！

file_<<"num1"<<num1;
输入的是标签，这里如果写成file_<<"num1："<<num1; 就会报错。
OpenCV Error: Bad argument (Key names may only contain alphanumeric characters [a-zA-Z0-9], '-', '_' and ' ') in icvYMLWrite


------
##FileNode 
// https://www.cnblogs.com/fourseas/p/5519681.html
/**
 *  FileStorage
    OpenCV 中的 FileStorage 类能够读写硬盘中的.xml和.yaml文件，这里我们只讨论对 .xml 的以下几种操作：
    写入（FileStorage::WRITE，覆盖写）
    追加（FileStorage::APPEND，追加写）
    读取（FileStorage::WRITE）
    FileStorage 以 FileNode 为单位存储数据，且无法删改某个已有 FileNode的内容，想实现删改功能，得自己造轮子……

    写入FileNode
    FileNode有两种类型，seq 和 map：
 */


FileStorage fs("data.xml", FileStorage::WRITE);
// seq_node 是一个 seq 型的节点, 以它为父节点，存入10个数据
fs << "seq_node" << "[";
for(size_t i = 0; i < 10; ++i){
  fs << i;
}
fs << "]";

// map_node 是一个 map 型节点, 以它为父节点，存入10个数据
fs << "map_node" << "{";
for(size_t i = 0; i < 10; ++i){
  fs << "node_" + to_string(i) << i;
}
fs << "}";

fs.release();
// 通过上面这段代码，我们可以看到 seq 和 map 这两种类型的节点，
// 在写入数据时的差别：前者在子节点间，写入一对方括号[], 而后者写入花括号{}；
// 前者在写入子节点的时候，无法为子节点命名，而后者可以。
// OpenCV 最重要的 Mat 类型在存储时是以 map 方式写入的。




// 读入FileNode
// seq 和 map 节点在读入数据的时候，
// 前者以索引的方式去获得子节点，后者用子节点的名字，
// 即一个字符串去获得子节点（字符串为键，节点为值）：
FileStorage fs("data.xml", FileStorage::READ);
vector<int> a, b;

// seq_node 是一个 seq 型的节点
FileNode seq_node = fs["seq_node"];
for(size_t i = 0; i < 10; ++i){
  seq_node[i] >> a[i];
}

// map_node 是一个 map 型节点
FileNode map_node = fs["map_node"];
for(size_t i = 0; i < 10; ++i){
  fs["node_" + to_string(i)] >> b[i];
}

fs.release();


// seq 型节点既然能以索引去取子节点，那自然有人会想到，
// 能不能用迭代器去访问子节点呢？答案是可以。
// OpenCV为我们提供了FileNode的迭代器：

// seq_node 是一个 seq 型的节点
FileNode seq_node = fs["seq_node"];
FileNodeIterator it = seq_node.begin();
for(; it != seq_node.end(); ++it){
  *it >> a[i];
}
自定义类型的读写
需要重载 write 和 read 函数：

struct MyData{
  int i;
  string str;
  Mat I;
}

// 自定义写入
void write(FileStorage &fs, const string &, const MyData &mydata){
  fs << "{"
     << "index" << mydata.i
     << "str"   << mydata.str
     << "img"   << mydata.I
     << "}"
}

// 自定义读取
void read(const FileNode &node, MyData &mydata, const MyData &default_val = MyData()){
  if(node.empty()) mydata = default_val;
  else {
    node["index"] >> mydata.i;
    node["str"] >> mydata.str;
    node["img"] >> mydata.I;
  }
}
