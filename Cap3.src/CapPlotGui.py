import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import os
import configparser
from types import SimpleNamespace
import sys
import matplotlib.pyplot as plt
from Utils.file_reader import read_file_with_named_columns,read_Alarms
from Utils.ProcessDataManual import ProcessCapPlotDataManual

#from Utils.data_utils import load_data

class CapPlotGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Capacity Plot GUI")
        self.root.geometry("800x300")
        self.root.minsize(400, 300)    # Easier to grab when not too small
        self.root.resizable(True, True)
        self.root.grid_columnconfigure(0, weight=0)  # Left zone
        self.root.grid_columnconfigure(1, weight=0)  # Mid  zone
        self.root.grid_columnconfigure(2, weight=1)  # Right zone    
        self.root.grid_rowconfigure(0, weight=1)   
        self.GUIZone = SimpleNamespace()    
        self.InitData()
        self.create_zones()  
        self.create_widgets()
        # GUI implementation
    def InitData(self):
        self.PLOTS = [None] * 50  # ?? may be no need of that
        self.open_figures = []
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        file_list_string = self.Get_From_Ini("Path", "FileList_POS")
        self.FileList_POS = [f.strip() for f in file_list_string.split(',')]
        
        file_list_string = self.Get_From_Ini("Path", "FileList_HDG")
        self.FileList_HDG = [f.strip() for f in file_list_string.split(',')]
        
        file_list_string = self.Get_From_Ini("Path", "FileList_SNS")
        self.FileList_SNS = [f.strip() for f in file_list_string.split(',')]

        file_list_string = self.Get_From_Ini("Path", "FileList_CTRL")
        self.FileList_CTRL = [f.strip() for f in file_list_string.split(',')]


        self.DataFolder = self.Get_From_Ini( "Path", "datafolder" )
        
        return

    def create_zones(self):
        print(f"Python executable: {sys.executable}")
        # Left zone - for buttons
        self.GUIZone.Z1 = tk.LabelFrame(self.root, text="Controls", 
                                        relief=tk.RIDGE, bd=2, padx=10, pady=10)
        self.GUIZone.Z1.place(relx=0.0, rely=0.0, relwidth=0.33, relheight=1.0)       
       # Mid zone - for file choose
        self.GUIZone.Z2 = tk.LabelFrame(self.root, text="FileList", 
                                        relief=tk.RIDGE, bd=2, padx=10, pady=10)
        self.GUIZone.Z2.place(relx=0.33, rely=0.0, relwidth=0.33, relheight=1.0)       

        # Right zone - for indicators
        self.GUIZone.Z3 = tk.LabelFrame(self.root, text="Status & Info", 
                                           relief=tk.RIDGE, bd=2, padx=10, pady=10)
        self.GUIZone.Z3.place(relx=0.66, rely=0.0, relwidth=0.33, relheight=1.0)    
        #============MidZone

    def create_widgets(self):
        # === LEFT ZONE - BUTTONS ===
            # Configure the button zone to have 10 rows
        # for i in range(10):
        #     self.GUIZone.Z1.grid_rowconfigure(i, weight=0)  # Fixed height rows

        # Button 1
        # self.root.update_idletasks()
        self.root.update()
        widthZ1 = self.GUIZone.Z1.winfo_width()
        heightZ1= self.GUIZone.Z1.winfo_height()
        self.btn1 = tk.Button(self.GUIZone.Z1, text="Calc DATA", 
                             command=lambda: self.Load_Ship_Cfg_callback("ID1"), 
                             bg="#1D431C", fg="white",state=tk.DISABLED)
        self.btn1.place(relx=0.01,rely=0.15,  width=150, height=30 )
        self.btn1.config(state=tk.NORMAL)
        # Button 2  
        self.btn2 = tk.Button(self.GUIZone.Z1, text="Load Data Folder", 
                             command=lambda: self.Load_Data_callback("ID2"), 
                             bg="#1D431C", fg="white",state=tk.DISABLED)
        self.btn2.place(relx=0.01,rely=0.00,  width=150, height=30 )
        self.btn2.config(state=tk.NORMAL)
        return 
        #============MidZone

    def on_closing(self):
        """Clean up all figures before closing main window"""
        # Close all matplotlib figures
        for fig in self.open_figures:
            plt.close(fig)
        
        # Clear the list
        self.open_figures.clear()
        
        # Destroy the main window
        self.root.destroy()
#--------------  End of initialisation ---------------------------------------
    def save_to_ini(self, SecN, ParN, ParVal):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ini_file = os.path.join(script_dir, "CapPlot.ini")
        config = configparser.ConfigParser()
        config.read(ini_file)
        config[SecN][ParN] = ParVal
        with open(ini_file, 'w') as f:
            config.write(f)
    def Get_From_Ini(self, SecN, ParN ):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ini_file = os.path.join(script_dir, "CapPlot.ini")
        config = configparser.ConfigParser()
        config.read(ini_file)
        ParVal=config[SecN][ParN] 
        return  ParVal


    def Load_Ship_Cfg_callback(self, ID):
        fig=ProcessCapPlotDataManual(self.DATA_SET)
        self.open_figures.extend(fig)
        return
    
    def Load_Data_callback(self, ID):
        folder_path = filedialog.askdirectory(  
            title="Choose DATA folder ",
            initialdir=self.DataFolder or os.getcwd() # Start in current directory
             )
        if not folder_path: 
            return
        self.save_to_ini("Path", "DataFolder", folder_path)
        self.DataFolder=folder_path
        
        self.DATA_SET = {   'POS': {},
                            'HDG': {},
                            'SNS': {},
                            'CTRL': {},
                             'ALRM':{}         }
        
        file_lists = {      'POS': self.FileList_POS,
                            'HDG': self.FileList_HDG,
                            'SNS': self.FileList_SNS,
                            'CTRL': self.FileList_CTRL    }
        
        for category, file_list in file_lists.items():
            for filename in file_list:
                full_path = os.path.join(self.DataFolder, filename)
                name_without_ext = os.path.splitext(filename)[0]
                try: 
                    data = read_file_with_named_columns(full_path)
                    self.DATA_SET[category][name_without_ext] = data
                    print(f"Loaded {category}/{name_without_ext}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    self.DATA_SET[category][name_without_ext] = None
               
        pos_data1 = self.DATA_SET['POS']['GGA0']
        TimeGGA=  pos_data1.get('Time')

        full_path = os.path.join(self.DataFolder, "Alarms.dat")
        Alarms=read_Alarms (full_path)
        self.DATA_SET['ALRM']['SMPL']=Alarms
        self.DATA_SET['ALRM']['FULL']=None
        return


if __name__ == "__main__":
    root = tk.Tk()
    app = CapPlotGui(root)
    root.mainloop()