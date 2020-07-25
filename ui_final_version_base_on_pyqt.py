import csv
import os
import qtawesome
from PyQt5 import QtWidgets, QtCore
import sys
from PyQt5.QtCore import QPoint, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QMouseEvent, QPixmap, QIcon


class Runthread_res(QtCore.QThread):
    #  通过类成员对象定义信号对象
    update = pyqtSignal(str)

    def __init__(self):
        super(Runthread_res, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        package_path = 'C:/Users/Lunatic/Anaconda3/envs/pytorch/Lib/site-packages'
        os.system("python predict_text.py %s" % package_path)
        res = '''
        '''
        self.update.emit(res)


class Runthread_ner(QtCore.QThread):
    #  通过类成员对象定义信号对象
    _signal = pyqtSignal(str)

    def __init__(self):
        super(Runthread_ner, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        package_path = 'C:/Users/Lunatic/Anaconda3/envs/pytorch/Lib/site-packages'
        os.system("python train_ner.py %s" % package_path)


class Runthread_link(QtCore.QThread):
    #  通过类成员对象定义信号对象
    _signal = pyqtSignal(str)

    def __init__(self):
        super(Runthread_link, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        package_path = 'C:/Users/Lunatic/Anaconda3/envs/pytorch/Lib/site-packages'
        os.system("python train_link.py %s" % package_path)


class MainUi(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(1500, 900)
        self.setWindowIcon(QIcon('./ui/title_icon.ico'))
        self.setWindowTitle('恒生电子')

        # 窗口主部件
        self.main_widget = QtWidgets.QWidget()
        self.main_widget.setObjectName('main_wideget')
        self.main_layout = QtWidgets.QGridLayout()
        self.main_widget.setLayout(self.main_layout)

        self.top_widget = QtWidgets.QWidget()
        self.top_widget.setObjectName('top_widget')
        self.top_layout = QtWidgets.QGridLayout()
        self.top_widget.setLayout(self.top_layout)
        op_top_widget = QtWidgets.QGraphicsOpacityEffect()
        op_top_widget.setOpacity(0.6)
        self.top_widget.setGraphicsEffect(op_top_widget)
        self.top_widget.setAutoFillBackground(True)

        # 顶部组件
        self.top_page = QtWidgets.QWidget()
        self.top_page.setObjectName('top_page')
        self.top_page_layout = QtWidgets.QGridLayout()
        self.top_page.setLayout(self.top_page_layout)

        # 小组信息
        self.ctrl_widget = QtWidgets.QWidget()
        self.ctrl_widget.setObjectName('ctrl_widget')
        self.ctrl_layout = QtWidgets.QGridLayout()
        self.ctrl_widget.setLayout(self.ctrl_layout)

        #  定义组件布局（在主组件上）
        self.main_layout.addWidget(self.ctrl_widget, 0, 0, 8, 10)
        self.setCentralWidget(self.main_widget)

        self.ctrl_info = QtWidgets.QWidget()
        self.ctrl_info.setObjectName('ctrl_info')
        self.ctrl_info_layout = QtWidgets.QGridLayout()
        self.ctrl_info.setLayout(self.ctrl_info_layout)
        op_ctrl_info = QtWidgets.QGraphicsOpacityEffect()
        op_ctrl_info.setOpacity(0.65)
        self.ctrl_info.setGraphicsEffect(op_ctrl_info)
        self.ctrl_info.setAutoFillBackground(True)

        self.ctrl_tra = QtWidgets.QWidget()
        self.ctrl_tra.setObjectName('ctrl_tra')
        self.ctrl_tra_layout = QtWidgets.QGridLayout()
        self.ctrl_tra.setLayout(self.ctrl_tra_layout)
        op_ctrl_tra = QtWidgets.QGraphicsOpacityEffect()
        op_ctrl_tra.setOpacity(0.65)
        self.ctrl_tra.setGraphicsEffect(op_ctrl_tra)
        self.ctrl_tra.setAutoFillBackground(True)

        self.ctrl_tra_res = QtWidgets.QWidget()
        self.ctrl_tra_res.setObjectName('ctrl_tra_res')
        self.ctrl_tra_res_layout = QtWidgets.QGridLayout()
        self.ctrl_tra_res.setLayout(self.ctrl_tra_res_layout)
        op_ctrl_tra_res = QtWidgets.QGraphicsOpacityEffect()
        op_ctrl_tra_res.setOpacity(0.65)
        self.ctrl_tra_res.setGraphicsEffect(op_ctrl_tra_res)
        self.ctrl_tra_res.setAutoFillBackground(True)

        self.ctrl_res = QtWidgets.QWidget()
        self.ctrl_res.setObjectName('ctrl_res')
        self.ctrl_res_layout = QtWidgets.QGridLayout()
        self.ctrl_res.setLayout(self.ctrl_res_layout)
        op_ctrl_res = QtWidgets.QGraphicsOpacityEffect()
        op_ctrl_res.setOpacity(0.65)
        self.ctrl_res.setGraphicsEffect(op_ctrl_res)
        self.ctrl_res.setAutoFillBackground(True)

        self.ctrl_show = QtWidgets.QWidget()
        self.ctrl_show.setObjectName('ctrl_show')
        self.ctrl_show_layout = QtWidgets.QGridLayout()
        self.ctrl_show.setLayout(self.ctrl_show_layout)
        op_ctrl_show = QtWidgets.QGraphicsOpacityEffect()
        op_ctrl_show.setOpacity(0.65)
        self.ctrl_show.setGraphicsEffect(op_ctrl_show)
        self.ctrl_show.setAutoFillBackground(True)

        self.ctrl_layout.addWidget(self.top_widget, 0, 0, 1, 6)
        self.ctrl_layout.addWidget(self.ctrl_info, 1, 2, 1, 4)
        self.ctrl_layout.addWidget(self.ctrl_tra, 1, 0, 2, 2)
        self.ctrl_layout.addWidget(self.ctrl_tra_res, 3, 0, 2, 2)
        self.ctrl_layout.addWidget(self.ctrl_res, 5, 0, 2, 2)
        self.ctrl_layout.addWidget(self.ctrl_show, 2, 2, 5, 4)

        # 选题信息
        self.top_label = QtWidgets.QLabel('             金融领域实体消歧')
        self.top_layout.addWidget(self.top_label)

        # 小组信息
        self.info_label = QtWidgets.QLabel('主创团队:')
        self.info_label.setObjectName('info_name_label')

        self.info_label_widget = QtWidgets.QWidget()
        self.info_label_widget.setObjectName('info_label_widget')
        self.info_label_layout = QtWidgets.QGridLayout()
        self.info_label_widget.setLayout(self.info_label_layout)
        op_info_label_widget = QtWidgets.QGraphicsOpacityEffect()
        op_info_label_widget.setOpacity(0.7)
        self.ctrl_res.setGraphicsEffect(op_info_label_widget)
        self.ctrl_res.setAutoFillBackground(True)

        self.ctrl_info_layout.addWidget(self.info_label, 0, 0, 1, 5)
        self.ctrl_info_layout.addWidget(self.info_label_widget, 0, 1, 1, 2)

        # 训练
        self.tra_label = QtWidgets.QLabel('训练：')
        self.tra_NER_1 = QtWidgets.QPushButton(qtawesome.icon('fa.adjust', color='red'), '实体识别训练')
        self.tra_NER_1.clicked.connect(self.start_ner)
        self.tra_WSD_1 = QtWidgets.QPushButton(qtawesome.icon('fa.search', color='orange'), '实体消歧训练')
        self.tra_WSD_1.clicked.connect(self.start_link)
        self.tra_space = QtWidgets.QPushButton()

        self.ctrl_tra_layout.addWidget(self.tra_label, 0, 0, 1, 3)
        self.ctrl_tra_layout.addWidget(self.tra_NER_1, 0, 1, 1, 3)
        self.ctrl_tra_layout.addWidget(self.tra_WSD_1, 2, 1, 1, 3)

        # 训练，只展示结果
        self.tra_res_label = QtWidgets.QLabel('结果：')
        self.tra_res_NER_1 = QtWidgets.QPushButton(qtawesome.icon('fa.user', color='#3cff00'), '识别验证结果')
        self.tra_res_NER_1.clicked.connect(self.click)
        self.tra_res_WSD_1 = QtWidgets.QPushButton(qtawesome.icon('fa.star', color='green'), '消歧验证结果')
        self.tra_res_WSD_1.clicked.connect(self.click)

        self.ctrl_tra_res_layout.addWidget(self.tra_res_label, 0, 0, 1, 3)
        self.ctrl_tra_res_layout.addWidget(self.tra_res_NER_1, 0, 1, 1, 3)
        self.ctrl_tra_res_layout.addWidget(self.tra_res_WSD_1, 2, 1, 1, 3)

        # 生成,同样只展示结果
        self.res_label = QtWidgets.QLabel('实例测试：')
        self.res_sub = QtWidgets.QPushButton(qtawesome.icon('fa.file', color='darkblue'), '消歧与结果保存')
        self.res_sub.clicked.connect(self.start_res)
        self.res_sub_sec = QtWidgets.QPushButton(qtawesome.icon('fa.database', color='purple'), '选择')
        self.res_sub_sec.clicked.connect(self.click)

        self.ctrl_res_layout.addWidget(self.res_label, 0, 0)
        # self.ctrl_res_layout.addWidget(self.res_cor, 1, 1)
        self.ctrl_res_layout.addWidget(self.res_sub, 1, 1)
        self.ctrl_res_layout.addWidget(self.res_sub_sec, 0, 1)

        # 展示界面
        self.show_title_label = QtWidgets.QLabel('')
        self.ctrl_show_layout.addWidget(self.show_title_label, 0, 0, 1, 3)
        self.show_label = QtWidgets.QLabel('')
        self.ctrl_show_layout.addWidget(self.show_label, 1, 0, 4, 5)
        pix = QPixmap('./ui/graph/wordcloud5.png')
        self.show_label.setPixmap(pix)

        # 展示各个分模型的测试结果
        self.show_form = QtWidgets.QLabel('')
        self.show_form.setVisible(False)

        # 查看识别的结果
        self.roberta_wwm_ner = QtWidgets.QPushButton('roberta-wwm')
        self.roberta_wwm_ner.clicked.connect(self.disp)
        self.roberta_wwm_ner.setVisible(False)
        self.wwm_ner = QtWidgets.QPushButton('wwm')
        self.wwm_ner.clicked.connect(self.disp)
        self.wwm_ner.setVisible(False)
        self.ernie_ner = QtWidgets.QPushButton('ernie')
        self.ernie_ner.clicked.connect(self.disp)
        self.ernie_ner.setVisible(False)
        self.return1 = QtWidgets.QPushButton('返回')
        self.return1.clicked.connect(self.disp)
        self.return1.setVisible(False)

        # 查看消歧的结果
        self.roberta_wwm_wsd = QtWidgets.QPushButton('roberta_wwm')
        self.roberta_wwm_wsd.clicked.connect(self.disp)
        self.roberta_wwm_wsd.setVisible(False)
        self.wwm_wsd = QtWidgets.QPushButton('wwm')
        self.wwm_wsd.clicked.connect(self.disp)
        self.wwm_wsd.setVisible(False)
        self.ernie_wsd = QtWidgets.QPushButton('ernie')
        self.ernie_wsd.clicked.connect(self.disp)
        self.ernie_wsd.setVisible(False)
        self.return2 = QtWidgets.QPushButton('返回')
        self.return2.clicked.connect(self.disp)
        self.return2.setVisible(False)

        self.show_res_label = QtWidgets.QLabel('')
        self.ctrl_show_layout.addWidget(self.show_res_label, 6, 0, 1, 5)

        # self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 隐藏边框
        self.main_layout.setSpacing(0)

        self.main_widget.setStyleSheet('''
                    QWidget#main_widget{
                    background:white;
                    }
        ''')
        # top_layout的美化
        self.top_widget.setStyleSheet('''
                    QWidget{
                        background:white;
                        border-top:0px solid gray;
                        border-bottom:0px solid gray;
                        border-left:0px solid gray;
                        border-radius:10px;
                        font-family: SimHei
                }
                QLabel{
                    font-size:70px;
                    text-align:center;
                }       
            ''')

        # ctrl_layout的美化
        self.ctrl_widget.setStyleSheet('''
                    #ctrl_widget{border-image:url(./ui/wallpaper/photo-1579532582937-16c108930bf6.jpg);
                    border-radius:10px;
                    font-family: SimHei
                    }
            ''')
        self.ctrl_info.setStyleSheet('''
                QWidget#ctrl_info{
                        background:white;
                        border-top:0px solid gray;
                        border-bottom:0px solid gray;
                        border-left:0px solid gray;
                        border-radius:10px;
                }
                QLabel{
                    font-size:42px;
                    font-family: SimHei
                } 
        ''')
        self.info_label_widget.setStyleSheet('''
                QWidget{
                    border-image:url(./ui/logo2.JPG);
                }
        ''')

        self.tra_space.setStyleSheet('''
                 QPushButton{
                        color:black;
                        font-size:30px;
                        height:40px;
                        padding-left:5px;
                        padding-right:5px;
                        text-align:center;
                        background:white;
                        border-radius:10px;
                    }

        ''')

        self.ctrl_tra.setStyleSheet('''
                    QLabel{
                        font-size:38px;
                    } 
                    QWidget{
                        background:white;
                        border-radius:10px;
                        font-family: SimHei
                    }
                    QPushButton{
                        color:black;
                        font-size:35px;
                        height:55px;
                        padding-left:4px;
                        padding-right:4px;
                        text-align:center;
                        background:#d4ff00;
                        border-radius:10px;
                    }
                    QPushButton:hover{
                        color:black;
                        border:1px solid #F3F3F5;
                        border-radius:10px;
                        background:orange;
                    }
                    QPushButton:pressed{
                        border-style:inset;  
                    }    
                ''')

        self.ctrl_tra_res.setStyleSheet('''
                            QLabel{
                                font-size:38px;
                                font-family: SimHei
                            } 
                            QWidget{
                                background:white;
                                border-radius:10px

                            }
                            QPushButton{
                                color:black;
                                font-size:35px;
                                height:55px;
                                padding-left:3px;
                                padding-right:3px;
                                text-align:center;
                                background:#ffe600;
                                border-radius:10px;
                                font-family: SimHei
                            }
                            QPushButton:hover{
                                color:black;
                                border:1px solid #F3F3F5;
                                border-radius:10px;
                                background:#ff8c00;
                            }    
                        ''')

        self.ctrl_res.setStyleSheet('''
                            QLabel{
                                font-size:38px;
                            } 
                            QWidget{
                                background:white;
                                border-radius:10px;
                                font-family: SimHei
                            }
                            QPushButton{
                                color:black;
                                font-size:35px;
                                height:55px;
                                padding-left:3px;
                                padding-right:3px;
                                text-align:center;
                                background:#ffae00;
                                border-radius:10px;
                            }
                            QPushButton:hover{
                                color:black;
                                border:1px solid #F3F3F5;
                                border-radius:10px;
                                background:#ff8c00;
                            }    
                        ''')
        self.show_title_label.setStyleSheet('''
                          QLabel{
                                color:black;
                                font-size:30px;
                                height:40px;
                                padding-left:5px;
                                padding-right:10px;
                                text-align:left;
                                border-radius:10px;
                                font-family: SimHei
                            }
                ''')
        self.show_res_label.setStyleSheet('''
                            QLabel{
                                color:red;
                                font-size:40px;
                                height:40px;
                                padding-left:200px;
                                padding-right:10px;
                                text-align:right;
                                border-radius:10px;
                                font-family: LiSu
                            }
                ''')
        self.ctrl_show.setStyleSheet('''
                    QWidget{
                            background:white;
                            background:white;
                            border-radius:10px;
                    }
                    QLabel{
                            color:#0033ff;
                            font-size:27px;
                            padding-left:10px;
                            padding-right:10px;
                            text-align:right;
                            border-radius:10px;
                            font-family: LiSu
                        }
                    QPushButton{
                        color:black;
                        font-size:30px;
                        height:35px;
                        padding-left:5px;
                        padding-right:5px;
                        text-align:center;
                        background:#ffe600;
                        border-radius:10px;
                    }
                    QPushButton:hover{
                                color:black;
                                border:1px solid #F3F3F5;
                                border-radius:10px;
                                background:#ff8c00;
                    }
                ''')

    def start_res(self):
        self.show_title_label.setText('正在生成……')
        self.show_label.setText('')
        self.show_res_label.setText('')
        # 创建线程
        self.thread = Runthread_res()
        # 开始线程
        self.thread.start()
        # 连接信号
        self.thread.update.connect(self.res_dis)

    def res_dis(self, res):
        self.show_title_label.setText('测试集消歧结果已生成')
        with open('./results_ner/bert/lstm_3_768_2/new_train_2_result.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            column = [row[3] for row in reader]

        self.show_label.setVisible(False)

        self.show_label_1 = QtWidgets.QLabel("")
        self.show_label_2 = QtWidgets.QLabel('')
        self.show_label_3 = QtWidgets.QLabel('')
        self.show_label_4 = QtWidgets.QLabel('')

        self.show_label_1.setVisible(True)
        self.ctrl_show_layout.addWidget(self.show_label_1, 1, 0, 1, 5)
        self.show_label_2.setVisible(True)
        self.ctrl_show_layout.addWidget(self.show_label_2, 2, 0, 1, 5)
        self.show_label_3.setVisible(True)
        self.ctrl_show_layout.addWidget(self.show_label_3, 3, 0, 1, 5)
        self.show_label_4.setVisible(True)
        self.ctrl_show_layout.addWidget(self.show_label_4, 4, 0, 1, 5)

        self.show_label_1.setText("""
语句："壹加壹表示,本次股权司法冻结不会对公司生产经营产生影响……
       """
        )
        self.show_label_2.setText("""
金融实体词:<font color='#f93e4a' size='+3' face='Sans'><b>壹加壹</b></font><br/><br/>
实体全称：<font color='#f93e4a' size='+0' face='Sans'><b>宁夏壹加壹农牧股份有限公司</b></font>
        """)
        self.show_label_3.setText("""
语句：#盘面# 水利板块直线拉升 安徽水利(600502)涨7.05%
        """)
        self.show_label_4.setText("""
金融实体词:<font color='#f9803e' size='+3' face='Sans'><b>安徽水利</b></font><br/><br/>
实体全称：<font color='#f9803e' size='+0' face='Sans'><b>安徽水利开发股份有限公司<b></font>
        """)
        self.show_res_label.setText('')

    def start_ner(self):
        # 创建线程
        self.thread = Runthread_ner()
        # 开始线程
        self.thread.start()

    def start_link(self):
        # 创建线程
        self.thread = Runthread_link()
        # 开始线程
        self.thread.start()

    def click(self):

        sender = self.sender()
        download_path = './data_deal/result.json'

        if sender == self.tra_res_NER_1:
            self.tra_res_label.setText('选择模型：')
            self.show_title_label.setText("识别验证：")
            self.tra_res_NER_1.setVisible(False)
            self.tra_res_WSD_1.setVisible(False)
            self.roberta_wwm_ner.setVisible(True)
            self.ctrl_tra_res_layout.addWidget(self.roberta_wwm_ner, 0, 2)
            self.wwm_ner.setVisible(True)
            self.ctrl_tra_res_layout.addWidget(self.wwm_ner, 1, 2)
            self.ernie_ner.setVisible(True)
            self.ctrl_tra_res_layout.addWidget(self.ernie_ner, 2, 2)
            self.return1.setVisible(True)
            self.ctrl_show_layout.addWidget(self.return1, 0, 4)
            self.tra_NER_1.setEnabled(False)
            self.tra_WSD_1.setEnabled(False)
            # self.tra_res_NER_1.setEnabled(False)
            # self.tra_res_WSD_1.setEnabled(False)
            self.res_sub.setEnabled(False)
            self.res_sub_sec.setEnabled(False)
            self.show_res_label.setVisible(True)
            self.show_title_label.setVisible(True)
            # pix = QPixmap('./ui/graph/train_ner_res_form.jpg')
            # self.show_label.setPixmap(pix)
            with open('./model_fuse/result.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                column = [row[3] for row in reader]
                self.show_res_label.setText("——模型融合结果:F1=%s——" % column[1])

        elif sender == self.tra_res_WSD_1:
            self.tra_res_label.setText('选择模型：')
            self.show_title_label.setText("消歧验证：")
            self.roberta_wwm_wsd.setVisible(True)
            self.ctrl_tra_res_layout.addWidget(self.roberta_wwm_wsd, 0, 2)
            self.wwm_wsd.setVisible(True)
            self.ctrl_tra_res_layout.addWidget(self.wwm_wsd, 1, 2)
            self.ernie_wsd.setVisible(True)
            self.ctrl_tra_res_layout.addWidget(self.ernie_wsd, 2, 2)
            self.return2.setVisible(True)
            self.ctrl_show_layout.addWidget(self.return2, 0, 4)
            self.show_res_label.setVisible(True)
            self.show_title_label.setVisible(True)
            self.tra_NER_1.setEnabled(False)
            self.tra_WSD_1.setEnabled(False)
            self.tra_res_NER_1.setVisible(False)
            self.tra_res_WSD_1.setVisible(False)
            self.res_sub.setEnabled(False)
            self.res_sub_sec.setEnabled(False)
            # pix = QPixmap('./ui/graph/train_link_res_form.jpg')
            # self.show_label.setPixmap(pix)
            with open('./model_fuse/result.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                column = [row[7] for row in reader]
                self.show_res_label.setText("——模型融合结果:F1=%s——" % column[1])

        elif sender == self.res_sub_sec:
            download_path = QtWidgets.QFileDialog.getOpenFileName(self, "浏览", "")  # 选取文件

    def disp(self):

        show = self.sender()

        if show == self.roberta_wwm_ner:
            pix = QPixmap('./results_ner/roberta_wwm/lstm_3_768_2/ner_roberta_wwm.jpg')
            self.show_label.setPixmap(pix)
            with open('./results_ner/roberta_wwm/lstm_3_768_2/roberta_wwm_new_train_2_result.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                column = [row[3] for row in reader]
                column.sort()
                self.show_res_label.setText("——当前模型最佳结果:F1=%s——" % column[len(column) - 2])

        elif show == self.wwm_ner:
            pix = QPixmap('./results_ner/wwm/lstm_3_768_2/ner_wwm.jpg')
            self.show_label.setPixmap(pix)
            with open('./results_ner/wwm/lstm_3_768_2/wwm_new_train_2_result.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                column = [row[3] for row in reader]
                column.sort()
                self.show_res_label.setText("——当前模型最佳结果:F1=%s——" % column[len(column) - 2])

        elif show == self.ernie_ner:
            pix = QPixmap('./results_ner/ernie/lstm_3_768_2/ner_ernie.jpg')
            self.show_label.setPixmap(pix)
            with open('./results_ner/ernie/lstm_3_768_2/new_train_2_result.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                column = [row[3] for row in reader]
                column.sort()
                self.show_res_label.setText("——当前模型最佳结果:F1=%s——" % column[len(column) - 2])

        elif show == self.return1:
            self.tra_res_label.setText('结果：')
            pix = QPixmap('./ui/graph/wordcloud5.png')
            self.show_label.setPixmap(pix)
            self.roberta_wwm_ner.setVisible(False)
            self.wwm_ner.setVisible(False)
            self.ernie_ner.setVisible(False)
            self.return1.setVisible(False)
            self.show_res_label.setText('')
            self.show_title_label.setText('')
            self.tra_NER_1.setEnabled(True)
            self.tra_WSD_1.setEnabled(True)
            self.tra_res_NER_1.setVisible(True)
            self.tra_res_WSD_1.setVisible(True)
            self.res_sub.setEnabled(True)
            self.res_sub_sec.setEnabled(True)

        elif show == self.roberta_wwm_wsd:
            pix = QPixmap('./results/roberta_wwm/lstm_3_768_2_len_400_lf_2_l_2/link_robert_wwm.jpg')
            self.show_label.setPixmap(pix)
            with open('./results/roberta_wwm/lstm_3_768_2_len_400_lf_2_l_2/new_train_2_result.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                column = [row[7] for row in reader]
                column.sort()
                self.show_res_label.setText("——当前模型最佳结果:F1=%s——" % column[len(column) - 2])

        elif show == self.wwm_wsd:
            pix = QPixmap('./results/wwm/lstm_3_768_2_len_400_lf_2_l_2/link_wwm.jpg')
            self.show_label.setPixmap(pix)
            with open('./results/wwm/lstm_3_768_2_len_400_lf_2_l_2/new_train_2_result.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                column = [row[7] for row in reader]
                column.sort()
                self.show_res_label.setText("——当前模型最佳结果:F1=%s——" % column[len(column) - 2])

        elif show == self.ernie_wsd:
            pix = QPixmap('./results/ernie/lstm_3_768_2_len_400_lf_2_l_2/link_ernie.jpg')
            self.show_label.setPixmap(pix)
            with open('./results/ernie/lstm_3_768_2_len_400_lf_2_l_2/new_train_2_result.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                column = [row[7] for row in reader]
                column.sort()
                self.show_res_label.setText("——当前模型最佳结果:F1=%s——" % column[len(column) - 2])

        elif show == self.return2:
            self.tra_res_label.setText('结果：')
            pix = QPixmap('./ui/graph/wordcloud5.png')
            self.show_label.setPixmap(pix)
            self.roberta_wwm_wsd.setVisible(False)
            self.wwm_wsd.setVisible(False)
            self.ernie_wsd.setVisible(False)
            self.return2.setVisible(False)
            self.show_res_label.setText('')
            self.show_title_label.setText('')
            self.tra_NER_1.setEnabled(True)
            self.tra_WSD_1.setEnabled(True)
            self.tra_res_NER_1.setVisible(True)
            self.tra_res_WSD_1.setVisible(True)
            self.res_sub.setEnabled(True)
            self.res_sub_sec.setEnabled(True)

    # 重写移动事件
    def mouseMoveEvent(self, e: QMouseEvent):
        self._endPos = e.pos() - self._startPos
        self.move(self.pos() + self._endPos)

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._isTracking = True
            self._startPos = QPoint(e.x(), e.y())

    def mouseReleaseEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._isTracking = False
            self._startPos = None
            self._endPos = None


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUi()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
