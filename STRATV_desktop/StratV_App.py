from tkinter import*
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
from tkinter.messagebox import*
from tkinter.filedialog import askopenfilename
import csv
import math
from decimal import*
import numpy as np
import random
import matplotlib
from scipy.optimize import*
import matplotlib.pyplot as plt
from pandastable.core import Table
from pandastable.data import TableModel
from datetime import datetime
import os
import sys

#=========================================================================================================================
# datetime object containing current date and time
now = datetime.now()

# dd/mm/YY H:M:S
date_time = now.strftime("%B %d, %Y | %H:%M:%S")
#print("date and time =", date_time)	
#==============================================================================================================================
#import os

#from idlelib.editor import EditorWindow
#def new(filename=None):
#    return EditorWindow(filename)
#=========================================================================================================================
#Reservoir and Process Data initialization and definiton.
    
Number_of_points = 0

Length_of_bed_ft = 0

width_of_bed_ft = 0

average_porosity = 0

VISO = 0

VISW = 0

OFVF = 0

WFVF = 0

SWI = 0

SGI = 0

SOI = 0

SOR = 0

#Constant Injection Rate in STB/D
Constant_injection_rate = 0

#Injection Pressure Differential in PSI
Inj_Pressure_differential = 0

Residual_gas_saturation_unswept_area = 0

Residual_gas_saturation_swept_area = 0

Residual_gas_saturation = Residual_gas_saturation_unswept_area+Residual_gas_saturation_swept_area

Saturation_gradient = 1-SOR-SWI

#=========================================================================================================================
root = Tk()
root.iconbitmap('STRATV.ico')
root.title('STRAT-V')

style = ttk.Style()
style.theme_use('clam')

# list the options of the style
# (Argument should be an element of TScrollbar, eg. "thumb", "trough", ...)
style.element_options("Horizontal.TScrollbar.thumb")

# configure the style
style.configure("Horizontal.TScrollbar", gripcount=0,
                background="#2196f3", darkcolor="#2196f3", lightcolor="grey",
                troughcolor="#2196f3", bordercolor="grey", arrowcolor="white")

#Create a main frame
main_frame = Frame(root,bg='#d6ebfb')
main_frame.pack(fill = BOTH, expand =1)

#Create a canvas

my_canvas = Canvas(main_frame,bg='#d6ebfb')
my_canvas.pack(side = LEFT, fill=BOTH, expand = 1)

#Add a scrollbar to the canvas
my_scrollbar = ttk.Scrollbar(main_frame, orient = VERTICAL, command = my_canvas.yview)
my_scrollbar.pack(side=RIGHT, fill = Y)
my_scrollbarx = ttk.Scrollbar(root, orient=HORIZONTAL, command=my_canvas.xview)
my_scrollbarx.pack(side=BOTTOM, fill = X)
#Configure the canvas
my_canvas.configure(xscrollcommand=my_scrollbarx.set, yscrollcommand = my_scrollbar.set)
my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion = my_canvas.bbox('all')))

#Create ANOTHER frame inside the canvas
second_frame = Frame(my_canvas, bg = '#c8f1f7')
Label(root, text = 'Copyright, 2020, Lekia Prosper').pack(side = BOTTOM)

#Add that frame to a windows in the canvas
my_canvas.create_window((0,0), window = second_frame, anchor = 'nw')        
#===========================================================================================================================
def Load_File():
    import_file=Tk()
    import_file.iconbitmap('STRATV.ico')
    import_file.title('STRAT-V')
    import_file.geometry('500x500')
    import_file.pack_propagate(False)
    #import_file.resizable(0,0)

    #frame for Treeview
    frame3=LabelFrame(import_file,text='Data File')
    #frame3.place(height=250, width=500)
    frame3.place(relheight=0.5, relwidth=1)

    #Frame for open filedialog
    file_frame=LabelFrame(import_file, text='Open File')
    file_frame.place(height=100, width=500, rely=0.65, relx=0)

    #Buttons
    button1=Button(file_frame, text = 'Browse A File',command=lambda:file_dialog())
    button1.place(rely=0.65, relx=0.8)

    button2= Button(file_frame, text = 'Load Permeability-Porosity Data', command=lambda: Load_Permeability_Porosity_Data())
    button2.place(rely=0.65,relx=0.4)
    button3=Button(file_frame, text = 'Load Relative Permeability Data', command=lambda: Load_Relative_Permeability_Data())
    button3.place(rely=0.65, relx=0)

    label_file = ttk.Label(file_frame, text = 'No File Selected')
    label_file.place(rely=0,relx=0)

    #Treeview Widget
    tv1=ttk.Treeview(frame3)
    tv1.place(relheight=1, relwidth=1)

    treescrolly = Scrollbar(frame3, orient='vertical',command=tv1.yview)
    treescrollx = Scrollbar(frame3, orient='horizontal',command=tv1.xview)
    tv1.configure(xscrollcommand=treescrollx.set)
    tv1.configure(yscrollcommand=treescrolly.set)
    treescrollx.pack(side='bottom', fill='x')
    treescrolly.pack(side='right', fill='y')

    def file_dialog():
        filename= filedialog.askopenfilename(initialdir="/", title = "Select A File", filetype=(("csvfiles","*.csv"),("All Files", "*.*")))
        label_file["text"]=filename

    def Load_Permeability_Porosity_Data():
        file_path=label_file['text']
        try:
            excel_filename=r"{}".format(file_path)
            bed_data=pd.read_csv(excel_filename)
        except ValueError:
            messagebox.showerror("Information","The file you have chosen is invalid")
            return None
        except FileNotFoundError:
            messagebox.showerror("Information", f"No such file as {file_path}")
            return None
        
        clear_data()
        tv1["column"]=list(bed_data.columns)
        tv1['show']='headings'
        for column in tv1['columns']:
            tv1.heading(column,text=column)
        bed_data_rows = bed_data.to_numpy().tolist()
        for row in bed_data_rows:
            tv1.insert("","end", values=row)
            
    def Load_Relative_Permeability_Data():
        file_path=label_file['text']
        try:
            excel_filename=r"{}".format(file_path)
            RPERM_data=pd.read_csv(excel_filename)
        except ValueError:
            messagebox.showerror("Information","The file you have chosen is invalid")
            return None
        except FileNotFoundError:
            messagebox.showerror("Information", f"No such file as {file_path}")
            return None

        clear_data()
        tv1["column"]=list(RPERM_data.columns)
        tv1['show']='headings'
        for column in tv1['columns']:
            tv1.heading(column,text=column)
        RPERM_data_rows = RPERM_data.to_numpy().tolist()
        for row in RPERM_data_rows:
            tv1.insert("","end", values=row)
        return None
    def clear_data():
        tv1.delete(*tv1.get_children())
    import_file.mainloop()
#===========================================================================================================================   
def openfile():
    filename = askopenfilename(parent=root)
    f = open(filename)
    f.read()
#===========================================================================================================================    
import pandas as pd
#============================================================================================================================
    
def fractional_flow():
    try:
        Fw_root = Tk()
        Fw_root.iconbitmap('STRATV.ico')
        Fw_root.title('STRAT-V')
        Fw_root.pack_propagate(False)
        Frame1 = LabelFrame(Fw_root, text='Result')
        Frame1.place(relheight=0.4, relwidth=0.6,rely=0, relx=0.05)
        listbox = Listbox(Frame1)
        listbox.place(relheight=0.8, relwidth=0.95,rely=0.1, relx=0.02)
    
        # Creating a Treeview
        Frame2 = LabelFrame(Fw_root, text='Fractional Flow Data')
        Frame2.place(relheight=0.5, relwidth=0.8,rely=0.5, relx=0.05)
        tree = ttk.Treeview(Frame2)
        tree.place(relheight=0.8, relwidth=0.95,rely=0.1, relx=0.02)
    
        treescrolly = Scrollbar(Frame2, orient='vertical',command=tree.yview)
        treescrollx = Scrollbar(Frame2, orient='horizontal',command=tree.xview)
        tree.configure(xscrollcommand=treescrollx.set)
        tree.configure(yscrollcommand=treescrolly.set)
        treescrollx.pack(side='bottom', fill='x')
        treescrolly.pack(side='right', fill='y')
    
        
        SW_table = pd.DataFrame(SW, columns = ['SW'])
        
    
        # Using the correlation between relative permeability ratio and water saturation
        import numpy as np
        np.seterr(divide='ignore', invalid='ignore')
        # Calculating the coefficient b
        b = (np.log((KRO/KRW)[2])-np.log((KRO/KRW)[3]))/(SW[3]-SW[2])
        
        #========================================================================
    
        # Calculating the coefficient a
        a = (KRO/KRW)[2]*math.exp(b*SW[2])
        
        #========================================================================
        # Calculating the fractional flow
        def fw(SW):
            fw = 1/(1+a*(VISW/VISO)*np.exp(-b*SW))
            return(fw)
        #========================================================================
        '''' To calculate a suitable slope for the tangent to the fractional flow curve
        Drawn from the initial water saturation'''
    
        # STEP1: Generate a list of uniformly distributed random numbers from a water saturation
        # greater than the initial water saturation to 1
        xList = []
        for i in range(0, 10000):
            x = random.uniform(SWI+0.1, 1)
            xList.append(x) 
        xs = np.array(xList)
    
        # STEP2: Calculate different slopes of tangents or lines intersecting the fractional
        # flow curve using the array generated in step 1 as the water saturation.
        m = 1/((xs-SWI)*(1+(VISW/VISO)*a*np.exp(-b*xs)))
    
        # STEP3: Calculate the maximum slope from different slopes generated in step 2.
        # The value of this slope will be the slope of the tangent to the fractional flow
        # curve.
        tangent_slope=max(m)
        #print('slope of the tangent line is:\n ',tangent_slope)
        #==========================================================================
        # Calculate the breakthrough saturation.
        Saturation_at_Breakthrough = SWI + 1/tangent_slope
        #print('saturation at breakthrough is:\n ', Saturation_at_Breakthrough)
        #===========================================================================
        # Calculating the saturation at the flood front
    
        def funct(SWF):
            swf = SWF[0]
            F = np.empty((1))
            F[0] = ((tangent_slope*(swf-SWI)*(1+(VISW/VISO)*a*math.exp(-b*swf)))-1)
            return F
        SWF_Guess = np.array([SWI+0.1])
        SWF = fsolve(funct, SWF_Guess)[0]
        SWF
        #============================================================================
        # Fractional flow at the flood front
        Fwf = fw(SWF)
        Fwf
        #=============================================================================
        # Fractional flow
        Fw = fw(SW)
        Fw_table = pd.DataFrame(Fw, columns = ['Fractional Flow (Fw)'])
        #print(Fw_table)
        #=============================================================================
        # Calculating the differential of the fractional flow equation
        dfw_dSw = (VISW/VISO)*a*b*np.exp(-SW*b)/(1+(VISW/VISO)*a*np.exp(-SW*b))**2
        dfw_dSw_table = pd.DataFrame(dfw_dSw, columns = ['dFw/dSw'])
        #print(dfw_dSw_table)
        #============================================================================
        # Generating the data for the tangent plot
        tangent = (SW-SWI)*tangent_slope
        tangent_table = pd.DataFrame(tangent, columns = ['Tangent'])
        #print(tangent_table)
        #============================================================================
        Fractional_flow_table = pd.concat([SW_table, Fw_table, dfw_dSw_table, tangent_table], axis=1)
        #print(Fractional_flow_table)
        #=============================================================================
        # Making the plots

        def plot():
            
            import pyqtgraph
            from PyQt5 import QtWidgets, QtCore
            from pyqtgraph import PlotWidget, plot
            import pyqtgraph as pg
            import sys  # We need sys so that we can pass argv to QApplication
            import os
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
            from matplotlib.figure import Figure
            class MainWindow(QtWidgets.QMainWindow):
            
                def __init__(self, *args, **kwargs):
                    super(MainWindow, self).__init__(*args, **kwargs)
            
                    self.graphWidget = pg.PlotWidget()
                    self.setCentralWidget(self.graphWidget)
            
                    #Add Background colour to white
                    self.graphWidget.setBackground('w')
                    # Add Title
                    self.graphWidget.setTitle("Fractional Flow Curve", color="b", size="20pt")
                    # Add Axis Labels
                    styles = {"color": "#f00", "font-size": "18px"}
                    self.graphWidget.setLabel("left", "Fractional Flow (Fw)", **styles)
                    self.graphWidget.setLabel("right", "Differential of Fractional Flow (dFw/dSw)", **styles)
                    self.graphWidget.setLabel("bottom", "Water Saturation (Sw)", **styles)
                    #Add legend
                    self.graphWidget.addLegend()
                    #Add grid
                    self.graphWidget.showGrid(x=True, y=True)
                    #Set Range
                    self.graphWidget.setXRange(0, 1, padding=0)
                    self.plot(SW, fw(SW), "Fw", 'r')
                    self.plot(SW, tangent, "Tangent", 'k')
                    self.plot(SW, dfw_dSw, "dFw/dSw", 'b')

            
                def plot(self, x, y, plotname, color):
                    pen = pg.mkPen(color=color)
                    self.graphWidget.plot(x, y, name=plotname, pen=pen, symbolBrush=(color))
            
            def main():
                app = QtWidgets.QApplication(sys.argv)
                main = MainWindow()
                main.show()
                main._exit(app.exec_())
                #QApplication.exec_()
            if __name__ == '__main__':
                main()
            
            
            
            
            
        #=========================================================================================================
        listbox.insert(1,"  Correlation: Kro/Krw = aexp(-bSw)") 
        listbox.insert(2, '  b : ' +str(b)) 
        listbox.insert(3, '  a : '+ str(a))
        listbox.insert(4, '  Slope of the tangent line : ' + str(tangent_slope)) 
        listbox.insert(5, '  Flood Front Saturation (Swf) : '+ str(SWF))
        listbox.insert(6, '  Flood Front Fractional Flow (Fwf) : '+ str(Fwf))
        listbox.insert(7, '  Saturation at breakthrough (SwBT) : '+ str(Saturation_at_Breakthrough)) 
    
        tree["column"]=list(Fractional_flow_table.columns)
        tree['show']='headings'
        for column in tree['columns']:
            tree.heading(column,text=column)
        Fractional_flow_table_rows = Fractional_flow_table.to_numpy().tolist()
        for row in Fractional_flow_table_rows:
            tree.insert("","end", values=row)
            
        View_plot = Button(Fw_root, text = 'View Plot',justify = LEFT,relief= RAISED,cursor='hand2',command = plot)   
        View_plot.place(rely=0.2, relx=0.7)
        #Quit_plot = Button(Fw_root, text = 'View Plot',justify = LEFT,relief= RAISED,cursor='hand2',command = sys.exit())   
        #Quit_plot.place(rely=0.5, relx=0.7)
        Fw_root.geometry("600x500")
    
        Fw_root.mainloop() 
    except ZeroDivisionError:
        return None

def enter_inputs():
    global entries
    global Number_of_points
    global Length_of_bed_ft
    global width_of_bed_ft
    global average_porosity
    global VISO
    global VISW
    global OFVF
    global WFVF
    global SWI
    global SGI
    global SOI
    global SOR
    global Constant_injection_rate
    global Inj_Pressure_differential
    global Residual_gas_saturation_unswept_area
    global Residual_gas_saturation_swept_area
    global Residual_gas_saturation
    
    fields = ('Number_of_points','Length_of_bed_ft', 'width_of_bed_ft', 'average_porosity','VISO', 'VISW','OFVF','WFVF','SWI', 'SGI','SOI','SOR','Constant_injection_rate',
    'Inj_Pressure_differential','Residual_gas_saturation_unswept_area','Residual_gas_saturation_swept_area','Residual_gas_saturation','Saturation_gradient')

    def residual_gas_saturation(entries):
        global RGSU
        global RGSS
        global RGS
        global Number_of_points
        global Length_of_bed_ft
        global width_of_bed_ft
        global average_porosity
        global VISO
        global VISW
        global OFVF
        global WFVF
        global SWI
        global SGI
        global SOI
        global SOR
        global Constant_injection_rate
        global Inj_Pressure_differential
        global Residual_gas_saturation_unswept_area
        global Residual_gas_saturation_swept_area
        global Residual_gas_saturation
        Number_of_points = float(entries['Number_of_points'].get())
        Length_of_bed_ft = float(entries['Length_of_bed_ft'].get())
        width_of_bed_ft = float(entries['width_of_bed_ft'].get())
        SWI = float(entries['SWI'].get())
        SOR = float(entries['SOR'].get())
        SGI = float(entries['SGI'].get())
        VISW = float(entries['VISW'].get())
        VISO = float(entries['VISO'].get())
        Saturation_gradient = float(entries['Saturation_gradient'].get())
        Constant_injection_rate = float(entries['Constant_injection_rate'].get())
        Inj_Pressure_differential = float(entries['Inj_Pressure_differential'].get())

        RGSU = float(entries['Residual_gas_saturation_unswept_area'].get())
        RGSS =  float(entries['Residual_gas_saturation_swept_area'].get())
        Residual_gas_saturation = float(entries['Residual_gas_saturation'].get())
        RGS = RGSU+RGSS
        RGS = ("%8.4f" % RGS).strip()
        entries['Residual_gas_saturation'].delete(0, END)
        entries['Residual_gas_saturation'].insert(0, RGS)
        
    def Saturation_gradient(entries):
        Saturation_gradient = float(entries['Saturation_gradient'].get())
        Saturation_gradient = 1-SOR-SWI
        Saturation_gradient = ("%8.4f" % Saturation_gradient).strip()
        entries['Saturation_gradient'].delete(0, END)
        entries['Saturation_gradient'].insert(0, Saturation_gradient)

    def makeform(inputs, fields):
        global entries
        entries = {}
        for field in fields:
           # print(field)
            row = Frame(inputs)
            lab = Label(row, width=22, text=field+": ", anchor='w')
            ent = Entry(row)
            ent.insert(0, "0")
            row.pack(side=TOP, 
                     fill=X, 
                     padx=5, 
                     pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, 
                     expand=YES,padx = 100, 
                     fill=X)
            entries[field] = ent
        return entries
    
    if __name__ == '__main__':
        inputs = Tk()
        inputs.iconbitmap('STRATV.ico')
        inputs.title('STRAT-V')
        ents = makeform(inputs, fields)
        b1 = Button(inputs, text='Residual gas saturation',
               command=(lambda e=ents:residual_gas_saturation(e)))
        b1.pack(side=LEFT, padx=5, pady=5)
        
        b2 = Button(inputs, text='Saturation gradient',
                   command=(lambda e1=ents:Saturation_gradient(e1)))
        b2.pack(side=LEFT, padx=5, pady=5)
        inputs.mainloop() 
Enter_input_button = Button(second_frame, text = 'Enter input data',justify = LEFT,relief= RAISED,cursor='hand2',command = enter_inputs).grid(row=2,column =0,padx=5,pady=10)    

#===============================================================================================================================           
Label(second_frame, text='Calculate',fg = 'white', bg = '#2196f3',justify = CENTER,relief= FLAT).grid(row=0, column=4, columnspan = 8,padx = 40,pady=10, sticky = W+E+N+S)
#Importing the Permeability Porosity distribution data
bed_data = pd.read_csv('Permeability_Porosity_distribution_data.csv')
#===========================================================================================================================      
#==========================================================================================================================
#Importing the Relative permeability Data
import pandas as pd
RPERM_data = pd.read_csv('Oil_Water_Relative_Permeability_data.csv')  

SW = np.array(RPERM_data['SW'])
KRW = np.array(RPERM_data['KRW'])
KRO = np.array(RPERM_data['KRO'])

#========================================================================================================================= 
def Reznik(entries):
    try:
        global RGSU
        global RGSS
        global RGS
        global Number_of_points
        global Length_of_bed_ft
        global width_of_bed_ft
        global average_porosity
        global VISO
        global VISW
        global OFVF
        global WFVF
        global SWI
        global SGI
        global SOI
        global SOR
        global Constant_injection_rate
        global Inj_Pressure_differential
        global Residual_gas_saturation_unswept_area
        global Residual_gas_saturation_swept_area
        global Residual_gas_saturation
        #Number_of_points = float(entries['Number_of_points'].get())
        Length_of_bed_ft = float(entries['Length_of_bed_ft'].get())
        width_of_bed_ft = float(entries['width_of_bed_ft'].get())
        SWI = float(entries['SWI'].get())
        SOI = float(entries['SOI'].get())
        SOR = float(entries['SOR'].get())
        SGI = float(entries['SGI'].get())
        VISW = float(entries['VISW'].get())
        VISO = float(entries['VISO'].get())
        OFVF = float(entries['OFVF'].get())
        WFVF = float(entries['WFVF'].get())
        Saturation_gradient = float(entries['Saturation_gradient'].get())
        Constant_injection_rate = float(entries['Constant_injection_rate'].get())
        Inj_Pressure_differential = float(entries['Inj_Pressure_differential'].get())

        #RGSU = float(entries['Residual_gas_saturation_unswept_area'].get())
        #RGSS =  float(entries['Residual_gas_saturation_swept_area'].get())

        import pandas as pd
        import math
        import numpy as np

        application_window = Tk()
        application_window.iconbitmap('STRATV.ico')
        application_window.title('STRAT-V')


        KRW_1_SOR = np.interp(1-SOR, SW, KRW)
        KRO_SWI = np.interp(SWI, SW, KRO)

        Water_Mobility = bed_data.PERMEABILITY*KRW_1_SOR/VISW
        Water_Mobility_table= pd.DataFrame(Water_Mobility).rename(columns={'PERMEABILITY': 'Water Mobility'})
        #Water_Mobility_table
        #==========================================================================================================================
        #CALCULATING THE WATER MOBILITY

        Oil_Mobility = bed_data.PERMEABILITY*KRO_SWI/VISO
        Oil_Mobility_table= pd.DataFrame(Oil_Mobility).rename(columns={'PERMEABILITY': 'Oil Mobility'})
        #Oil_Mobility_table
        #=========================================================================================================================
        # CALCULATING THE MOBILITY RATIO, M.
        #import math
        Mobility_Ratio = Water_Mobility/Oil_Mobility
        Mobility_Ratio_table= pd.DataFrame(Mobility_Ratio).rename(columns={'PERMEABILITY': 'MOBILITY RATIO'})
        #Mobility_Ratio_table

        #==========================================================================================================================

        # ARRANGING THE DATA IN ORDER OF DECREASING PERMEABILITY.
        Bed_ordering_parameter=np.array(bed_data.POROSITY)*Saturation_gradient*(1+Mobility_Ratio)/Water_Mobility

        # ARRANGING THE DATA IN ORDER OF DECREASING PERMEABILITY.
        #Bed Ordering
        Bed_ordering_parameter = np.array(bed_data.POROSITY)*Saturation_gradient*(1+Mobility_Ratio)/Water_Mobility
        Bed_ordering_parameter_table = pd.DataFrame(Bed_ordering_parameter).rename(columns={'PERMEABILITY': 'BED ORDERING PARAMETER'})
        bed_data_combine = pd.concat([bed_data, Bed_ordering_parameter_table,Water_Mobility_table,Oil_Mobility_table,Mobility_Ratio_table], axis = 1)

        bed_data_sort = bed_data_combine.sort_values(by='BED ORDERING PARAMETER',ignore_index=True, ascending=True) 
        Average_porosity = '%.3f' % np.mean(bed_data_sort.POROSITY)
        #===========================================================================================================================
        # Extracting input variables from data table.
        import numpy as np
        Layers = np.array(bed_data_sort['LAYER'])
        Layer_table1 = pd.DataFrame(Layers, columns=['Layers'])
        Bed_ordering_parameter = np.array(bed_data_sort['BED ORDERING PARAMETER'])
        Bed_ordering_parameter_sort_table = pd.DataFrame(Bed_ordering_parameter)
        PORO = np.array(bed_data_sort['POROSITY'])
        Porosity_sort_table = pd.DataFrame(PORO)
        permeability_array = np.array(bed_data_sort['PERMEABILITY'])

        Water_mobility_array = np.array(bed_data_sort['Water Mobility'])
        Water_mobility_sort_table = pd.DataFrame(Water_mobility_array)

        Oil_mobility_array = np.array(bed_data_sort['Oil Mobility'])
        Oil_mobility_sort_table = pd.DataFrame(Oil_mobility_array)

        Permeability_sort_table = pd.DataFrame(permeability_array)
        bed_thickness = np.array(bed_data_sort['THICKNESS'])

        #==========================================================================================================================

        #Bed order parameter ratio of each bed to the last bed
        bed_order_ratio_list = []
        for j in range(len(Layers)):
                bed_order_ratio_to_lastbed = bed_data_sort['BED ORDERING PARAMETER'][j]/bed_data_sort['BED ORDERING PARAMETER'].iat[-1]
                bed_order_ratio_list.append(bed_order_ratio_to_lastbed)
        bed_order_ratio=pd.DataFrame(bed_order_ratio_list)

        #==========================================================================================================================

        #Bed order parameter ratio of each bed to the last bed
        bed_order_ratio_to_other_beds_list = []
        for j in range(len(Layers)):
                bed_order_ratio_to_otherbeds = bed_data_sort['BED ORDERING PARAMETER'].iat[-1]/bed_data_sort['BED ORDERING PARAMETER'][j]
                bed_order_ratio_to_other_beds_list.append(bed_order_ratio_to_otherbeds)
        bed_order_ratio_to_otherbeds=pd.DataFrame(bed_order_ratio_to_other_beds_list)

        #===========================================================================================================================
        #Flood front position of bed n when bed j has just broken through.

        last_mobility_ratio = bed_data_sort['MOBILITY RATIO'].iat[-1]

        Flood_front_position_of_bed_n_j = (-last_mobility_ratio+np.sqrt(last_mobility_ratio**2+(bed_order_ratio)*(1-last_mobility_ratio**2)))/(1-last_mobility_ratio)
        Flood_front_position_of_bed_n_j = pd.DataFrame(Flood_front_position_of_bed_n_j).rename(columns = {0:'Flood Front Position of the last bed at breakthrough of other beds'})

        #==========================================================================================================================
        global Number_of_points
        #Flood front location of the last bed.
        Number_of_points = float(entries['Number_of_points'].get())
        #converting the flood front table to a list
        flood_front_of_last_bed = 0
        flood_front_of_last_bed_list = []
        if Mobility_Ratio_table.iloc[0,0] == 1:
            Flood_front_position_of_bed_n_j_list=bed_order_ratio.to_list()
            for index, position in list(enumerate(Flood_front_position_of_bed_n_j_list)):
                while flood_front_of_last_bed < Flood_front_position_of_bed_n_j_list[0]:
                   # if flood_front_of_last_bed < Flood_front_position_of_bed_n_j_list[0]:
                    flood_front_of_last_bed = flood_front_of_last_bed + Flood_front_position_of_bed_n_j_list[0]/Number_of_points
                    flood_front_of_last_bed_list.append(flood_front_of_last_bed)


                if(index > 0):
                    while flood_front_of_last_bed >=Flood_front_position_of_bed_n_j_list[index-1] and flood_front_of_last_bed <= Flood_front_position_of_bed_n_j_list[index]:
                        flood_front_of_last_bed = flood_front_of_last_bed + (Flood_front_position_of_bed_n_j_list[index]-Flood_front_position_of_bed_n_j_list[index-1])/Number_of_points
                        flood_front_of_last_bed_list.append(flood_front_of_last_bed)

        else:
            Flood_front_position_of_bed_n_j_list=Flood_front_position_of_bed_n_j['Flood Front Position of the last bed at breakthrough of other beds'].to_list()
            for index, position in list(enumerate(Flood_front_position_of_bed_n_j_list)):
                while flood_front_of_last_bed < Flood_front_position_of_bed_n_j_list[0]:
                   # if flood_front_of_last_bed < Flood_front_position_of_bed_n_j_list[0]:
                    flood_front_of_last_bed = flood_front_of_last_bed + Flood_front_position_of_bed_n_j_list[0]/Number_of_points
                    flood_front_of_last_bed_list.append(flood_front_of_last_bed)


                if(index > 0):
                    while flood_front_of_last_bed >=Flood_front_position_of_bed_n_j_list[index-1] and flood_front_of_last_bed <= Flood_front_position_of_bed_n_j_list[index]:
                        flood_front_of_last_bed = flood_front_of_last_bed + (Flood_front_position_of_bed_n_j_list[index]-Flood_front_position_of_bed_n_j_list[index-1])/Number_of_points
                        flood_front_of_last_bed_list.append(flood_front_of_last_bed)


        flood_front_of_last_bed_table = pd.DataFrame(flood_front_of_last_bed_list).rename(columns = {0:'Flood Front Position of the last bed at time t'})

        #===========================================================================================================================

        #Calculating Real or Process time for the CIP case
        porosity_of_last_bed = bed_data_sort['POROSITY'].iat[-1]
        water_mobility_of_last_bed = bed_data_sort['Water Mobility'].iat[-1]
        Real_time_CIP = 158.064*((Length_of_bed_ft**2/Inj_Pressure_differential)*porosity_of_last_bed*Saturation_gradient/water_mobility_of_last_bed)*(last_mobility_ratio*np.array(flood_front_of_last_bed_list) + 0.5*(1-last_mobility_ratio)*np.array(flood_front_of_last_bed_list)**2)
        Real_time_CIP_table = pd.DataFrame(Real_time_CIP).rename(columns = {0:'Real time for constant injection pressure'})
        #Real_time_CIP_table

        #==========================================================================================================================
        # Calculating breakthrough time of each bed.
        porosity_of_last_bed = bed_data_sort['POROSITY'].iat[-1]
        water_mobility_of_last_bed = bed_data_sort['Water Mobility'].iat[-1]
        breakthrough_time = 158.064*((Length_of_bed_ft**2/Inj_Pressure_differential)*porosity_of_last_bed*Saturation_gradient/water_mobility_of_last_bed)*(last_mobility_ratio*np.array(Flood_front_position_of_bed_n_j['Flood Front Position of the last bed at breakthrough of other beds'].to_list()) + 0.5*(1-last_mobility_ratio)*np.array(Flood_front_position_of_bed_n_j['Flood Front Position of the last bed at breakthrough of other beds'].to_list())**2)
        breakthrough_time_table = pd.DataFrame(breakthrough_time).rename(columns = {0:'Breakthrough time'})
        #breakthrough_time_table

        #==========================================================================================================================
        # Flood front position of other beds with resect to bed n
        Flood_front_location_of_other_beds_list = []
        for j in range(len(Layers)):
            aj = Mobility_Ratio[j]**2
            bed_order_of_last_bed = bed_data_sort['BED ORDERING PARAMETER'].iat[-1]
            bj = (bed_order_ratio_to_other_beds_list[j])*(2*last_mobility_ratio/(1+last_mobility_ratio))*(1-Mobility_Ratio[j]**2)
            cj = (bed_order_ratio_to_other_beds_list[j])*((1-last_mobility_ratio)/(1+last_mobility_ratio))*(1-Mobility_Ratio[j]**2)

            Flood_front_location_of_other_beds = (-Mobility_Ratio[j]+np.sqrt(aj+bj*np.array(flood_front_of_last_bed_list)+cj*np.array(flood_front_of_last_bed_list)**2))/(1-Mobility_Ratio[j])

            Flood_front_location_of_other_beds_list.append(Flood_front_location_of_other_beds)
            for i in range(len(Flood_front_location_of_other_beds_list[j])):
                if Flood_front_location_of_other_beds_list[j][i] > 1:
                     Flood_front_location_of_other_beds_list[j][i] = 1 
        Flood_front_location_of_other_beds_table = pd.DataFrame(Flood_front_location_of_other_beds_list).transpose()
        #Flood_front_location_of_other_beds_table
        #==========================================================================================================================
                
        
        #==========================================================================================================================
        '''# Front position of other beds at breakthrough.'''
        Front_position_of_other_beds_at_breakthrough_list = []
        
        if Mobility_Ratio_table.iloc[0,0] == 1:
            for j in range(len(Layers)):
                Front_position_of_other_beds_at_breakthrough= ((bed_order_ratio_to_other_beds_list[j])*((1-last_mobility_ratio)*(np.array(Flood_front_position_of_bed_n_j['Flood Front Position of the last bed at breakthrough of other beds'].to_list()))**2 +2*last_mobility_ratio*(np.array(Flood_front_position_of_bed_n_j['Flood Front Position of the last bed at breakthrough of other beds'].to_list()))))/(1+last_mobility_ratio)
                Front_position_of_other_beds_at_breakthrough_list.append(Front_position_of_other_beds_at_breakthrough)
        else:    
            for j in range(len(Layers)):
                aj = Mobility_Ratio[j]**2
                bed_order_of_last_bed = bed_data_sort['BED ORDERING PARAMETER'].iat[-1]
                bj = (bed_order_ratio_to_other_beds_list[j])*(2*last_mobility_ratio/(1+last_mobility_ratio))*(1-Mobility_Ratio[j]**2)
                cj = (bed_order_ratio_to_other_beds_list[j])*((1-last_mobility_ratio)/(1+last_mobility_ratio))*(1-Mobility_Ratio[j]**2)
        
                Front_position_of_other_beds_at_breakthrough = (-Mobility_Ratio[j]+np.sqrt(aj+bj*np.array(Flood_front_position_of_bed_n_j['Flood Front Position of the last bed at breakthrough of other beds'].to_list())+cj*np.array(Flood_front_position_of_bed_n_j['Flood Front Position of the last bed at breakthrough of other beds'].to_list())**2))/(1-Mobility_Ratio[j])
        
                Front_position_of_other_beds_at_breakthrough_list.append(Front_position_of_other_beds_at_breakthrough)
        
        Front_position_of_other_beds_at_breakthrough_table = pd.DataFrame(Front_position_of_other_beds_at_breakthrough_list)
        #============================================================================================
        
        # Flood front position of other beds with resect to bed n. This is to know how far each front has advanced beyond the bed
        from decimal import Decimal
        Flood_front_location_of_other_beds_beyond_breakthrough_list = []
        if Mobility_Ratio_table.iloc[0,0] == 1:
            for j in range(len(Layers)):
                Flood_front_location_of_other_beds_beyond_breakthrough= ((bed_order_ratio_to_other_beds_list[j])*((1-last_mobility_ratio)*(flood_front_of_last_bed_list)**2 +2*last_mobility_ratio*(flood_front_of_last_bed_list)))/(1+last_mobility_ratio)
                Flood_front_location_of_other_beds_beyond_breakthrough_list.append(Flood_front_location_of_other_beds_beyond_breakthrough)
        else:
            for j in range(len(Layers)):
                aj = Mobility_Ratio[j]**2
                bed_order_of_last_bed = bed_data_sort['BED ORDERING PARAMETER'].iat[-1]
                bj = (bed_order_ratio_to_other_beds_list[j])*(2*last_mobility_ratio/(1+last_mobility_ratio))*(1-Mobility_Ratio[j]**2)
                cj = (bed_order_ratio_to_other_beds_list[j])*((1-last_mobility_ratio)/(1+last_mobility_ratio))*(1-Mobility_Ratio[j]**2)
    
                Flood_front_location_of_other_beds_beyond_breakthrough = (-Mobility_Ratio[j]+np.sqrt(aj+bj*np.array(flood_front_of_last_bed_list)+cj*np.array(flood_front_of_last_bed_list)**2))/(1-Mobility_Ratio[j])
    
                Flood_front_location_of_other_beds_beyond_breakthrough_list.append(Flood_front_location_of_other_beds_beyond_breakthrough)
        Flood_front_location_of_other_beds_beyond_breakthrough_table = pd.DataFrame(Flood_front_location_of_other_beds_beyond_breakthrough_list).transpose().round(4)
        #Flood_front_location_of_other_beds_beyond_breakthrough_table

        #==================================================================================================================================
        Property_time_list = []
        for i in range(len(Layers)):
            Property_time = 158.064*((Length_of_bed_ft**2/Inj_Pressure_differential)*bed_data_sort['POROSITY'][i]*Saturation_gradient/bed_data_sort['Water Mobility'][i])*(Mobility_Ratio[i]*Flood_front_location_of_other_beds_beyond_breakthrough_table[i]+0.5*(1-Mobility_Ratio[i])*Flood_front_location_of_other_beds_beyond_breakthrough_table[i]**2)
            Property_time_list.append(Property_time)
        Property_time_table= pd.DataFrame(Property_time_list).T
        #Property_time_table

        #==========================================================================================================================
        #Average mobility of the fluids in each bed at time t
        average_mobility_at_time_t_list = []
        for i in range(len(Layers)):
            average_mobility_at_time_t = Water_mobility_array[i]/(Mobility_Ratio[i]+(1-Mobility_Ratio[i])*Flood_front_location_of_other_beds_beyond_breakthrough_table[i])
            average_mobility_at_time_t_list.append(average_mobility_at_time_t)
        average_mobility_at_time_t_table = pd.DataFrame(average_mobility_at_time_t_list).transpose()

        #==========================================================================================================================
        #Superficial filter velocity of Darcy's law at time t
        Superficial_filter_velocity_list = []
        for i in range(len(Layers)):
            Superficial_filter_velocity = (Inj_Pressure_differential/Length_of_bed_ft)*average_mobility_at_time_t_table[i]
            Superficial_filter_velocity_list.append(Superficial_filter_velocity)
        Superficial_filter_velocity_table = pd.DataFrame(Superficial_filter_velocity_list).transpose()

        #==========================================================================================================================
        #Real time actual linear velocity of the flood front.
        actual_linear_velocity_list = []
        for i in range(len(Layers)):
            actual_linear_velocity  = Superficial_filter_velocity_table[i]/(bed_data_sort['POROSITY'][i]*Saturation_gradient)
            actual_linear_velocity_list.append(actual_linear_velocity)
        actual_linear_velocity_table = pd.DataFrame(actual_linear_velocity_list).transpose()

        #==========================================================================================================================
        # Instantaneous volumetric flow rate of water into bed.
        instantaneous_volumetric_flowrate_of_water_list = []
        for i in range(len(Layers)):
            instantaneous_volumetric_flowrate_of_water = 0.0011267*width_of_bed_ft*bed_thickness[i]*Superficial_filter_velocity_table[i]
            instantaneous_volumetric_flowrate_of_water_list.append(instantaneous_volumetric_flowrate_of_water)
        instantaneous_volumetric_flowrate_of_water_table = pd.DataFrame(instantaneous_volumetric_flowrate_of_water_list).transpose()
        #instantaneous_volumetric_flowrate_of_water_table

        #==========================================================================================================================
        # Instantaneous volumetric flow rate of oil into bed.
        instantaneous_volumetric_flowrate_of_oil_list = []
        for i in range(len(Layers)):
            instantaneous_volumetric_flowrate_of_oil = 0.0011267*width_of_bed_ft*bed_thickness[i]*Superficial_filter_velocity_table[i]/((1-bed_data_sort['MOBILITY RATIO'][i])*Flood_front_location_of_other_beds_table[i]+bed_data_sort['MOBILITY RATIO'][i])
            instantaneous_volumetric_flowrate_of_oil_list.append(instantaneous_volumetric_flowrate_of_oil)
        instantaneous_volumetric_flowrate_of_oil_table = pd.DataFrame(instantaneous_volumetric_flowrate_of_oil_list).transpose()

        #==========================================================================================================================
        # Total flow rate for each bed.
        Constant_total_injection_rate_list = []
        for i in range(len(Layers)):
            Constant_total_injection_rate = np.sum(instantaneous_volumetric_flowrate_of_water_table[i])
            Constant_total_injection_rate_list.append(Constant_total_injection_rate)
        Constant_total_injection_rate_table = pd.DataFrame(Constant_total_injection_rate_list)
        Constant_total_injection_rate_for_all_beds = Constant_total_injection_rate_table.sum(axis=0).values[0]
        #Constant_total_injection_rate_for_all_beds
        #==========================================================================================================================
        # Get the count of ones in each column at a given time
        number_of_ones_list = {}
        for i in range(len(Real_time_CIP_table)+1):
            number_of_ones_list[i] = Flood_front_location_of_other_beds_table[0:i].isin([1]).sum().to_frame().T.iloc[0,:]
            #number_of_ones_list.append(number_of_ones)
        number_of_ones_table = pd.DataFrame.from_dict(number_of_ones_list).T

        # returns the column with lowest count of 1 at a given time period. this represents the dynamic bed.
        number_of_ones_table['Dynamic_bed'] = number_of_ones_table.idxmin(axis=1)
        pd.set_option("display.max_rows", None, "display.max_columns", None)

        #==========================================================================================================================
        dynamic_bed = number_of_ones_table['Dynamic_bed']
        dynamic_bed_table = pd.DataFrame(dynamic_bed).rename(columns = {'Dynamic_bed':'Dynamic Bed'})
        #dynamic_bed_table = pd.DataFrame(number_of_ones_table['Dynamic_bed'], columns = ['Dynamic bed'])
        water_flow_rate_and_Dynamic_bed=pd.concat([instantaneous_volumetric_flowrate_of_water_table,dynamic_bed], axis = 1)
        #print(dynamic_bed_table)
        #==========================================================================================================================
        #just before breakthrough of the dynamic bed
        sum_water_flowrate_before_breakthrough_of_dynamic_bed_list = []
        for i in range(len(Real_time_CIP_table)):
            for j in range(len(Layers)):
                if dynamic_bed[i] == j:
                    sum_water_flowrate_before_breakthrough_of_dynamic_bed = instantaneous_volumetric_flowrate_of_water_table.iloc[i,0:j].sum(axis = 0)
                    sum_water_flowrate_before_breakthrough_of_dynamic_bed_list.append(sum_water_flowrate_before_breakthrough_of_dynamic_bed)
        sum_water_flowrate_before_breakthrough_of_dynamic_bed_table = pd.DataFrame(sum_water_flowrate_before_breakthrough_of_dynamic_bed_list).rename(columns = {0:'Sum water flowrate before breakthrough of dynamic bed'})
        #sum_water_flowrate_before_breakthrough_of_dynamic_bed_table

        #==========================================================================================================================
        #just before breakthrough of the dynamic bed
        sum_oil_flowrate_before_breakthrough_of_dynamic_bed_list = []
        for i in range(len(Real_time_CIP_table)):
            for j in range(len(Layers)):
                if dynamic_bed[i] == j:
                    sum_oil_flowrate_before_breakthrough_of_dynamic_bed = instantaneous_volumetric_flowrate_of_oil_table.iloc[i,j:len(Layers)+1].sum(axis = 0)
                    sum_oil_flowrate_before_breakthrough_of_dynamic_bed_list.append(sum_oil_flowrate_before_breakthrough_of_dynamic_bed)
        sum_oil_flowrate_before_breakthrough_of_dynamic_bed_table = pd.DataFrame(sum_oil_flowrate_before_breakthrough_of_dynamic_bed_list).rename(columns = {0:'Sum oil flowrate before breakthrough of dynamic bed'})
        sum_oil_flowrate_before_breakthrough_of_dynamic_bed_table

        #==========================================================================================================================
        # Instantaneous producing WOR, defined at xj = l, for all j, at time t just before breakthrough of the dynamic bed
        Instantaneous_producing_Water_Oil_Ratio_before_breakthrough_of_dynamic_bed = (np.array(sum_water_flowrate_before_breakthrough_of_dynamic_bed_list)/np.array(sum_oil_flowrate_before_breakthrough_of_dynamic_bed_list))
        Instantaneous_producing_Water_Oil_Ratio_before_breakthrough_of_dynamic_bed_table = pd.DataFrame(Instantaneous_producing_Water_Oil_Ratio_before_breakthrough_of_dynamic_bed).rename(columns = {0:'Instantaneous producing Water Oil ratio before breakthrough of dynamic bed'})
        #Instantaneous_producing_Water_Oil_Ratio_before_breakthrough_of_dynamic_bed_table

        # Instantaneous producing Water cut, defined at xj = l, for all j, at time t just before breakthrough of the dynamic bed
        Instantaneous_producing_Water_cut = np.array(sum_water_flowrate_before_breakthrough_of_dynamic_bed_list)/(np.array(sum_oil_flowrate_before_breakthrough_of_dynamic_bed_list)+np.array(sum_water_flowrate_before_breakthrough_of_dynamic_bed_list))
        Instantaneous_producing_Water_cut_table = pd.DataFrame(Instantaneous_producing_Water_cut).rename(columns = {0:'Instantaneous producing Water cut'})
        #Instantaneous_producing_Water_cut_table

        #==========================================================================================================================
        # Ultimate recoverable oil per bed
        global Ultimate_recoverable_oil_per_bed_table
        Ultimate_recoverable_oil_per_bed_list = []
        for i in range(len(Layers)):
            Ultimate_recoverable_oil_per_bed = 0.1781*Length_of_bed_ft*width_of_bed_ft*bed_data_sort['THICKNESS'][i]*bed_data_sort['POROSITY'][i]*Saturation_gradient
            Ultimate_recoverable_oil_per_bed_list.append(Ultimate_recoverable_oil_per_bed)
        Ultimate_recoverable_oil_per_bed_table = pd.DataFrame(Ultimate_recoverable_oil_per_bed_list).rename(columns = {0:'Ultimate recoverable oil per bed'})
        #Ultimate_recoverable_oil_per_bed_table
        #==========================================================================================================================
        # Total recoverable oil in place for the entire system of n beds.

        Total_recoverable_oil_in_place = Ultimate_recoverable_oil_per_bed_table.sum(axis = 0).values[0]

        #Total_recoverable_oil_in_place

        #==========================================================================================================================
        # Product of flood front location and ultimate recovery at per bed.
        Product_of_flood_front_location_and_ultimate_recovery_list = []
        for j in range(len(Layers)):
            Product_of_flood_front_location_and_ultimate_recovery = Flood_front_location_of_other_beds_beyond_breakthrough_table[j]*Ultimate_recoverable_oil_per_bed_table.iloc[j,0]
            Product_of_flood_front_location_and_ultimate_recovery_list.append(Product_of_flood_front_location_and_ultimate_recovery)
        Product_of_flood_front_location_and_ultimate_recovery_table = pd.DataFrame(Product_of_flood_front_location_and_ultimate_recovery_list).T
        #Product_of_flood_front_location_and_ultimate_recovery_table

        #==========================================================================================================================
        # cumulative oil recovered from all beds at time t .
        # Term 1
        cumulative_oil_recovered_at_time_t_list = []
        for i in range(len(Real_time_CIP_table)):
            for j in range(len(Layers)):
                if dynamic_bed[i] == j:
                    cumulative_oil_recovered_at_time_t = np.array(Ultimate_recoverable_oil_per_bed_list)[0:j].sum(axis = 0)
                    cumulative_oil_recovered_at_time_t_list.append(cumulative_oil_recovered_at_time_t)
        cumulative_oil_recovered_at_time_t_table = pd.DataFrame(cumulative_oil_recovered_at_time_t_list)
        #cumulative_oil_recovered_at_time_t_table

        # Term 2
        cumulative_oil_recovered_and_flood_front_location_at_time_t_list = []
        for k in range(len(Real_time_CIP_table)):
            for l in range(len(Layers)):
                if dynamic_bed[k] == l:
                    cumulative_oil_recovered_and_flood_front_location_at_time_t = Product_of_flood_front_location_and_ultimate_recovery_table.iloc[k,l:len(Layers)+1].sum(axis = 0)
                    cumulative_oil_recovered_and_flood_front_location_at_time_t_list.append(cumulative_oil_recovered_and_flood_front_location_at_time_t)
        cumulative_oil_recovered_and_flood_front_location_at_time_t_table = pd.DataFrame(cumulative_oil_recovered_and_flood_front_location_at_time_t_list)
        #cumulative_oil_recovered_and_flood_front_location_at_time_t_table

        # Cumulative oil recovered from all beds at time t
        Cumulative_oil_recovered_from_all_beds = cumulative_oil_recovered_at_time_t_table + cumulative_oil_recovered_and_flood_front_location_at_time_t_table
        Cumulative_oil_recovered_from_all_beds_table = pd.DataFrame(Cumulative_oil_recovered_from_all_beds).rename(columns = {0:'Cumulative oil recovered from all beds at time t'})
        
        #Cumulative_oil_recovered_from_all_beds_table
        #=========================================================================================================================
        # Vertical coverage at time t
        Vertical_coverage_at_time_t = Cumulative_oil_recovered_from_all_beds/Total_recoverable_oil_in_place
        Vertical_coverage_at_time_t_table = pd.DataFrame(Vertical_coverage_at_time_t).rename(columns = {0:'Vertical coverage at time t'})
        #Vertical_coverage_at_time_t_table

        #=========================================================================================================================
        # Cumumlative water oil ratio for constant injecction rate case.
        Cumumlative_water_oil_ratio_for_CIR = ((Constant_total_injection_rate_for_all_beds*Real_time_CIP_table['Real time for constant injection pressure']) - Cumulative_oil_recovered_from_all_beds_table['Cumulative oil recovered from all beds at time t'])/Cumulative_oil_recovered_from_all_beds_table['Cumulative oil recovered from all beds at time t']
        Cumumlative_water_oil_ratio_for_CIR_table = pd.DataFrame(Cumumlative_water_oil_ratio_for_CIR).rename(columns = {0:'Cumumlative water oil ratio for constant injection rate'})
        #Cumumlative_water_oil_ratio_for_CIR_table
        #=========================================================================================================================
        # Cumumlative water oil ratio for constant injecction Pressure case.
        # First get the product of difference between the real time and the breakthrough time, the bed thickness and water mobility.
        product_1_list  = []
        for j in range(len(Layers)):
            product_1 = (Real_time_CIP_table['Real time for constant injection pressure'].to_numpy() - breakthrough_time_table['Breakthrough time'][j])*bed_data_sort['THICKNESS'][j]*instantaneous_volumetric_flowrate_of_water_table[j].to_numpy()
            product_1_list.append(product_1)
        product_1_table = pd.DataFrame(product_1_list).T
        #product_1_table

        Cumumlative_water_oil_ratio_for_CIP_list = []
        for i in range(len(Real_time_CIP_table)):
            for j in range(len(Layers)):
                if dynamic_bed[i] == j:
                    Cumumlative_water_oil_ratio_for_CIP = ((width_of_bed_ft*Inj_Pressure_differential/Length_of_bed_ft)*product_1_table.iloc[i, 0:j].sum(axis = 0))/Cumulative_oil_recovered_from_all_beds_table['Cumulative oil recovered from all beds at time t'][i]
                    Cumumlative_water_oil_ratio_for_CIP_list.append(Cumumlative_water_oil_ratio_for_CIP)
        Cumumlative_water_oil_ratio_for_CIP_table = pd.DataFrame(Cumumlative_water_oil_ratio_for_CIP_list).rename(columns = {0:'Cumumlative water oil ratio for constant injection pressure'})
        #Cumumlative_water_oil_ratio_for_CIP_table
        #=========================================================================================================================
        # The cumulative water injected into bed i to time t, is given for the Constant injection pressure case by;
        cumulative_water_injected_list = []
        #for i in range(len(Real_time_CIP_table)):
        for j in range(len(Layers)):
            cumulative_water_injected_1 = Flood_front_location_of_other_beds_beyond_breakthrough_table[j]*Ultimate_recoverable_oil_per_bed_table['Ultimate recoverable oil per bed'][j]
            cumulative_water_injected_2 = Ultimate_recoverable_oil_per_bed_table['Ultimate recoverable oil per bed'][j] + 1.1267e-3*(width_of_bed_ft*Inj_Pressure_differential/Length_of_bed_ft)*product_1_table[j]
            #cumulative_water_injected_list_1.append(cumulative_water_injected_1)
            for i in range(len(Real_time_CIP_table)):
                if Real_time_CIP_table['Real time for constant injection pressure'][i] <= breakthrough_time[j]:

                    cumulative_water_injected_list.append(cumulative_water_injected_1)
                    #cumulative_water_injected = Flood_front_location_of_other_beds_beyond_breakthrough_table.iloc[:,j]*Ultimate_recoverable_oil_per_bed_table[0][j]

                else:
                    cumulative_water_injected_list.append(cumulative_water_injected_2)
                break
               # cumulative_water_injected = Ultimate_recoverable_oil_per_bed_table[0][j] + (width_of_bed_ft*Inj_Pressure_differential/Length_of_bed_ft)*product_1_table[j]

        cumulative_water_injected_table = pd.DataFrame(cumulative_water_injected_list).T
        cumulative_water_injected_table


        #==========================================================================================================================

        global General
        global Flood_front_location_of_other_beds_table_time
        global Front_position_of_other_beds_at_breakthrough_table_time
        global Flood_front_location_of_other_beds_beyond_breakthrough_table_time
        global average_mobility_at_time_t_table_time
        global Superficial_filter_velocity_table_time
        global actual_linear_velocity_table_time
        global instantaneous_volumetric_flowrate_of_water_table_time
        global instantaneous_volumetric_flowrate_of_oil_table_time

        # TABLE OF ALL OBTAINED VALUES.
        General =pd.concat([Layer_table1,breakthrough_time_table,Flood_front_position_of_bed_n_j,Ultimate_recoverable_oil_per_bed_table,
                            Front_position_of_other_beds_at_breakthrough_table.rename(columns=lambda x: str(x)+' Flood_Front_position_of_beds_at_breakthrough'),
                            flood_front_of_last_bed_table,Real_time_CIP_table,dynamic_bed_table,sum_water_flowrate_before_breakthrough_of_dynamic_bed_table,
                            sum_oil_flowrate_before_breakthrough_of_dynamic_bed_table,Instantaneous_producing_Water_Oil_Ratio_before_breakthrough_of_dynamic_bed_table,
                            Instantaneous_producing_Water_cut_table,cumulative_oil_recovered_at_time_t_table,Cumulative_oil_recovered_from_all_beds_table,
                            Vertical_coverage_at_time_t_table,Cumumlative_water_oil_ratio_for_CIR_table,Cumumlative_water_oil_ratio_for_CIP_table,
                            Flood_front_location_of_other_beds_table.rename(columns=lambda x: str(x)+' Flood_front_location_of_beds'),
                            Flood_front_location_of_other_beds_beyond_breakthrough_table.rename(columns=lambda x: str(x)+' Flood_front_location_of_beds_beyond_breakthrough'),
                            Property_time_table.rename(columns=lambda x: str(x)+' Property Time'),
                            average_mobility_at_time_t_table.rename(columns=lambda x: str(x)+' Average_mobility'),
                            Superficial_filter_velocity_table.rename(columns=lambda x: str(x)+' Superficial_filter_velocity'),
                            actual_linear_velocity_table.rename(columns=lambda x: str(x)+' Actual_linear_velocity'),
                            instantaneous_volumetric_flowrate_of_water_table.rename(columns=lambda x: str(x)+' Instantaneous_volumetric_flowrate_of_water'),
                            instantaneous_volumetric_flowrate_of_oil_table.rename(columns=lambda x: str(x)+' Instantaneous_volumetric_flowrate_of_oil'),
                            cumulative_water_injected_table.rename(columns=lambda x: str(x)+' Cumulative Water Injected')
                            ],axis=1)
    except ZeroDivisionError:
        messagebox.showerror("Omission", "Enter a value of Number of points other than zero")
        return None
    except _tkinter.TclError:
        return None
    #====================================================================================================================================================
   
    def general():

        class MyTable(Table):
            
            def __init__(self, parent=None, **kwargs):
                Table.__init__(self, parent, **kwargs)
                return

        class MyApp(Frame):

            def __init__(self, parent=None):
                self.parent = parent
                Frame.__init__(self)
                #self.main = self.master
                self.application_window = self.master
                #self.main.geometry('800x600+200+100')
                self.application_window.title('Reznik et al General data and graph')
                f = Frame(self.application_window)
                f.pack(side=RIGHT,expand=1)
                pt = make_table(f)
                bp = Frame(self.application_window)
                bp.pack(side=TOP)
            
                return

        def make_table(table_frame1, **kwds):
            
            df = General
            pt = MyTable(table_frame1, dataframe=df, **kwds )
            pt.show()
            return pt
        def test1():
            """just make a table"""

            t = Toplevel()
            fr = Frame(t)
            fr.pack(fill=BOTH, expand=1)
            pt = make_table(fr)
            return

        def select_test():
            """cell selection and coloring"""

            #t = Toplevel()
            #fr = Frame(t)
            fr = table_frame1
            #fr.pack(fill=BOTH, expand=1)
            pt = Table(fr)
            pt.show()
            pt.General
            pt.resetIndex(ask=False)
            pt.columncolors = {'c':'#fbf1b8'}
            df = pt.model.df
            

            mask_1 = df.a<7
            pt.setColorByMask('a', mask_1, '#337ab7')
            colors = {'red':'#f34130','blue':'blue'}
            for l in df.label.unique():
                mask = df['label']==l
                pt.setColorByMask('label', mask, l) 
            pt.redraw()
            return

        def multiple_tables():
            """make many tables in one frame"""

            t = Toplevel(height=800)
            r=0;c=0
            for i in range(6):
                fr = Frame(t)
                fr.grid(row=r,column=c)
                pt = make_table(fr, showtoolbar=True, showstatusbar=True)
                c+=1
                if c>2:
                    c=0
                    r+=1
            return
            
        app = MyApp()
        app.mainloop()
    
    #===========================================================================================================================
    table_frame1 = Frame(application_window)
    table_frame1.place(relheight = 1, relwidth =1)
    Button(table_frame1, text = 'View data',bg = '#337ab7',fg = 'white',justify = LEFT,relief= RAISED,cursor='hand2',command = general).grid(row=2,column =2,padx=5,pady=5,stick = W)    
    
    #================================================================================================================================
    application_window.geometry("200x150")
    application_window.mainloop() 
#================================================================================================


#===============================================================================================================================        

# Extracting input variables from data table.

# ARRANGING THE DATA IN ORDER OF DECREASING PERMEABILITY.
bed_data_sort = bed_data.sort_values(by='PERMEABILITY', ascending=False)

PORO = np.array(bed_data_sort['POROSITY'])
permeability_array = np.array(bed_data_sort['PERMEABILITY'])
h = np.array(bed_data_sort['THICKNESS'])
SW = np.array(RPERM_data['SW'])
KRW = np.array(RPERM_data['KRW'])
KRO = np.array(RPERM_data['KRO'])
#==========================================================================================================================
#This code calculates the permeability ratio, ki/kn
List_of_permeability_ratio = []
for permeability_index in range(len(permeability_array)):
    List_of_permeability_ratio_subset = [][:-permeability_index]
    for index,permeability in enumerate(permeability_array):
        if permeability_index <= index:
            permaebility_ratio = permeability/permeability_array[permeability_index]
            List_of_permeability_ratio_subset.append(permaebility_ratio)
    List_of_permeability_ratio.append(List_of_permeability_ratio_subset)

List_of_permeability_ratio_DataTable = pd.DataFrame(List_of_permeability_ratio).transpose()
#==========================================================================================================================
KRW_1_SOR = np.interp(1-SOR, SW, KRW)
KRO_SWI = np.interp(SWI, SW, KRO)

# Calculating the average porosity
def average_porosity():
    Average_porosity = '%.3f' % np.mean(bed_data_sort.POROSITY)
    Label(second_frame, text= str(Average_porosity),justify = LEFT, relief = SUNKEN).grid(row = 3, column = 6,padx = 40,pady=5, sticky =W)
Button(second_frame,text='Average Porosity',bg = '#337ab7',fg = 'white',cursor='hand2', command=average_porosity).grid(row=2,column=6,padx = 40,pady=10, sticky =W)

def relative_perm_1_SOR(entries):
    SOR = float(entries['SOR'].get())
    KRW_1_SOR = '%.3f' % np.interp(1-SOR, SW, KRW)
    Label(second_frame, text=  str(KRW_1_SOR),justify = LEFT, relief = SUNKEN).grid(row = 3, column = 7,padx = 40,pady=5, sticky =W)
    #return KRW_1_SOR
Button(second_frame,text='Relative Permeability at 1-SOR',bg = '#337ab7',fg = 'white',cursor='hand2',command=(lambda: relative_perm_1_SOR(entries))).grid(row=2,column=7,padx = 40,pady=10, sticky =W)    

def relative_perm_SWI(entries):
    SWI = float(entries['SWI'].get())
    KRO_SWI = '%.3f' % np.interp(SWI, SW, KRO)
    Label(second_frame, text= str(KRO_SWI),justify = LEFT, relief = SUNKEN).grid(row = 5, column = 6,padx = 40,pady=5, sticky =W)
    #return KRO_SWI
Button(second_frame,text='Relative Permeability at Initial Water Saturation',bg = '#337ab7',fg = 'white',cursor='hand2',command=(lambda: relative_perm_SWI(entries))).grid(row=4,column=6,padx = 40,pady=10, sticky =W)

def mobility_ratio(entries):
    SWI = float(entries['SWI'].get())
    VISW = float(entries['VISW'].get())
    VISO = float(entries['VISO'].get())
    SOR = float(entries['SOR'].get())
    KRW_1_SOR = np.interp(1-SOR, SW, KRW)
    KRO_SWI = np.interp(SWI, SW, KRO)
    Mobility_Ratio =  KRW_1_SOR*VISO/(KRO_SWI*VISW)
    Label(second_frame, text= str(Mobility_Ratio),justify = LEFT, relief = SUNKEN).grid(row = 7, column = 7,padx = 40,pady=5, sticky =W)
    #return Mobility_Ratio
Button(second_frame,text='Mobility Ratio',bg = '#337ab7',fg = 'white',cursor='hand2',command=(lambda: mobility_ratio(entries))).grid(row=6,column=7,padx = 40,pady=10, sticky =W)

def areal_sweep_efficiency_at_breakthrough(entries):
    SWI = float(entries['SWI'].get())
    VISW = float(entries['VISW'].get())
    VISO = float(entries['VISO'].get())
    SOR = float(entries['SOR'].get())
    KRW_1_SOR = np.interp(1-SOR, SW, KRW)
    KRO_SWI = np.interp(SWI, SW, KRO)
    Mobility_Ratio = KRW_1_SOR*VISO/(KRO_SWI*VISW)
    Areal_sweep_efficiency_at_breakthrough =  0.54602036+(0.03170817/Mobility_Ratio)+(0.30222997/math.exp(Mobility_Ratio)-0.0050969*Mobility_Ratio)
    Label(second_frame, text= str(Areal_sweep_efficiency_at_breakthrough),justify = LEFT,relief = SUNKEN).grid(row = 9, column = 6,padx = 40,pady=5, sticky =W)
    #return Areal_sweep_efficiency_at_breakthrough
Button(second_frame,text='Areal sweep efficiency at breakthrough',bg = '#337ab7',fg = 'white',cursor='hand2',command=(lambda: areal_sweep_efficiency_at_breakthrough(entries))).grid(row=8,column=6,padx = 40,pady=10, sticky =W)

def area_acres(entries):
    global Length_of_bed_ft
    global width_of_bed_ft
    Length_of_bed_ft = float(entries['Length_of_bed_ft'].get())
    width_of_bed_ft = float(entries['width_of_bed_ft'].get())
    Area_acres =  Length_of_bed_ft*width_of_bed_ft/43560
    Label(second_frame, text= str(Area_acres)+ ' acres',justify = LEFT,relief = SUNKEN).grid(row = 9, column = 7,padx = 40,pady=5, sticky =W)
    #return Area_acres
Button(second_frame,text='Area of the reservoir bed',bg = '#337ab7',fg = 'white',cursor='hand2',command=(lambda: area_acres(entries))).grid(row=8,column=7,padx = 40,pady=10, sticky =W)

def gross_rock_volume(entries):
    global Length_of_bed_ft
    global width_of_bed_ft
    Length_of_bed_ft = float(entries['Length_of_bed_ft'].get())
    width_of_bed_ft = float(entries['width_of_bed_ft'].get())
    Area_acres = Length_of_bed_ft*width_of_bed_ft/43560
    Gross_rock_volume_acre_ft =  Area_acres*bed_data_sort.THICKNESS.sum()
    Label(second_frame, text= str(Gross_rock_volume_acre_ft)+ ' acres-ft',justify = LEFT,relief = SUNKEN).grid(row = 5, column = 7,padx = 40,pady=5, sticky =W)
    #return Gross_rock_volume_acre_ft
Button(second_frame,text='Gross rock volume',bg = '#337ab7',fg = 'white',cursor='hand2',command=(lambda: gross_rock_volume(entries))).grid(row=4,column=7,padx = 40,pady=10, sticky =W)

def displacement_efficiency(entries):
    global SGI
    global SWI
    global SOR
    global Displacement_efficiency
    SWI = float(entries['SWI'].get())
    SOR = float(entries['SOR'].get())
    SGI = float(entries['SGI'].get())
    Displacement_efficiency =  (1-SWI-SGI-SOR)/(1-SWI-SGI)
    Label(second_frame, text= str(Displacement_efficiency),justify = LEFT,relief = SUNKEN).grid(row = 7, column = 6,padx = 40,pady=5, sticky =W)
    #return Displacement_efficiency
Button(second_frame,text='Displacement efficiency',bg = '#337ab7',fg = 'white',cursor='hand2',command=(lambda: displacement_efficiency(entries))).grid(row=6,column=6,padx = 40,pady=10, sticky =W)

def areal_sweep_efficiency(entries):
    global SGI
    global SWI
    global SOR
    global SGI
    global Constant_injection_rate 
    global Inj_Pressure_differential
    SWI = float(entries['SWI'].get())
    SOR = float(entries['SOR'].get())
    SGI = float(entries['SGI'].get())
    VISW = float(entries['VISW'].get())
    VISO = float(entries['VISO'].get())
    Constant_injection_rate = float(entries['Constant_injection_rate'].get())
    Inj_Pressure_differential = float(entries['Inj_Pressure_differential'].get())
    KRW_1_SOR = np.interp(1-SOR, SW, KRW)
    KRO_SWI = np.interp(SWI, SW, KRO)
    Mobility_Ratio = KRW_1_SOR*VISO/(KRO_SWI*VISW)
    Areal_sweep_efficiency_at_breakthrough = 0.54602036+(0.03170817/Mobility_Ratio)+(0.30222997/math.exp(Mobility_Ratio)-0.0050969*Mobility_Ratio)
    Displacement_efficiency = (1-SWI-SGI-SOR)/(1-SWI-SGI)
    Areal_sweep_efficiency =  Areal_sweep_efficiency_at_breakthrough+0.2749*np.log((1/Displacement_efficiency))
    Label(second_frame, text= str(Areal_sweep_efficiency),justify = LEFT,relief = SUNKEN).grid(row = 13, column = 6,padx = 40,pady=10, sticky =W)
    #return Areal_sweep_efficiency
Button(second_frame,text='Areal sweep efficiency',bg = '#337ab7',fg = 'white',cursor='hand2',command=(lambda: areal_sweep_efficiency(entries))).grid(row=12,column=6,padx = 40,pady=10, sticky =W)
#============================================================================================================================================================



# EXTRACTING THE SORTED LAYER COLUMN
Layer_column = bed_data_sort['LAYER'].to_numpy()
Layer_table =  pd.DataFrame(Layer_column, columns = ['Layers'])
#==================================================================================================================================================================
#Calculating the oil mobility ratio
#==========================================================================================================================
#This code calculates the list of waterflood front location as each layer breaksthrough
def D_tables(entries):
    global All_tables
    global RGSU
    global RGSS
    global RGS
    global Number_of_points
    global Length_of_bed_ft
    global width_of_bed_ft
    global average_porosity
    global VISO
    global VISW
    global OFVF
    global WFVF
    global SWI
    global SGI
    global SOI
    global SOR
    global Constant_injection_rate
    global Inj_Pressure_differential
    global Residual_gas_saturation_unswept_area
    global Residual_gas_saturation_swept_area
    global Residual_gas_saturation
    Number_of_points = float(entries['Number_of_points'].get())
    Length_of_bed_ft = float(entries['Length_of_bed_ft'].get())
    width_of_bed_ft = float(entries['width_of_bed_ft'].get())
    SWI = float(entries['SWI'].get())
    SOR = float(entries['SOR'].get())
    SOI = float(entries['SOI'].get())
    SGI = float(entries['SGI'].get())
    VISW = float(entries['VISW'].get())
    VISO = float(entries['VISO'].get())
    OFVF = float(entries['OFVF'].get())
    WFVF = float(entries['WFVF'].get())
    Saturation_gradient = float(entries['Saturation_gradient'].get())
    Constant_injection_rate = float(entries['Constant_injection_rate'].get())
    Inj_Pressure_differential = float(entries['Inj_Pressure_differential'].get())
    
    import pandas as pd
    import math
    import numpy as np
    
    D_window = Tk()
    D_window.iconbitmap('STRATV.ico')
    D_window.title('STRAT-V')

    KRW_1_SOR = np.interp(1-SOR, SW, KRW)
    KRO_SWI = np.interp(SWI, SW, KRO)
    Oil_Mobility =  permeability_array*KRO_SWI/VISO
    Oil_Mobility_table = pd.DataFrame(Oil_Mobility, columns = ['Oil Mobility'])

    Water_Mobility =  permeability_array*KRW_1_SOR/VISW
    Water_Mobility_table = pd.DataFrame(Water_Mobility, columns = ['Water Mobility'])

    Front_Location_list = []
    Mobility_Ratio = KRW_1_SOR*VISO/(KRO_SWI*VISW)
    Areal_sweep_efficiency_at_breakthrough = 0.54602036+(0.03170817/Mobility_Ratio)+(0.30222997/math.exp(Mobility_Ratio)-0.0050969*Mobility_Ratio)
    Displacement_efficiency = (1-SWI-SGI-SOR)/(1-SWI-SGI)
    import numpy as np
    Areal_sweep_efficiency =  Areal_sweep_efficiency_at_breakthrough+0.2749*np.log((1/Displacement_efficiency))
    
    #Average_porosity = np.mean(bed_data_sort.POROSITY)
    Average_porosity = np.mean(bed_data_sort.POROSITY)
    Area_acres = Length_of_bed_ft*width_of_bed_ft/43560
    Gross_rock_volume_acre_ft =  Area_acres*bed_data_sort.THICKNESS.sum()
    
    for permeability_index1 in range(len(permeability_array)):
        Front_Location = (Mobility_Ratio - np.sqrt(Mobility_Ratio**2+List_of_permeability_ratio_DataTable[permeability_index1]*(1-Mobility_Ratio**2)))/(Mobility_Ratio-1)
        Front_Location_list.append(Front_Location)
    #This code generates table of flood front location as the layers breakthrough
    Front_Location_list_DataTable = pd.DataFrame(Front_Location_list).transpose()
    #==========================================================================================================================
    # CALCULATING THE OIL FLOW RATE IN EACH BED AS EACH BED BREAKS THROUGH
    Water_Flowrate_per_bed = (0.0011267*width_of_bed_ft*bed_data_sort['THICKNESS']*Inj_Pressure_differential/Length_of_bed_ft)*Water_Mobility

    Water_Flowrate_per_bed_table = pd.DataFrame(Water_Flowrate_per_bed).rename(columns={'THICKNESS':'Water Flowrate Per Bed (Barrels/D)'})
    #=============================================================================================================================
    Water_Flowrate_list = []
    for n in range(len(permeability_array)):
        Water_Flowrate = Water_Flowrate_per_bed_table.iloc[0:n].sum()
        Water_Flowrate_list.append(Water_Flowrate)
    Water_Flowrate_table = pd.DataFrame(Water_Flowrate_list).rename(columns={'Water Flowrate Per Bed (Barrels/D)':'Water Production Rate (Barrels/D)'})
    #===========================================================================================================================
    # CALCULATING THE OIL FLOW RATE IN EACH BED AS EACH BED BREAKS THROUGH
    Oil_Flowrate_per_bed_list = []
    for bed in Front_Location_list_DataTable.columns:
        Oil_Flowrate_per_bed = (0.0011267*width_of_bed_ft*bed_data_sort['THICKNESS']*Inj_Pressure_differential/Length_of_bed_ft)*Water_Mobility/((1-Mobility_Ratio)*Front_Location_list_DataTable[bed]+Mobility_Ratio)
        Oil_Flowrate_per_bed_list.append(Oil_Flowrate_per_bed)    
    Oil_Flowrate_per_bed_table = pd.DataFrame(Oil_Flowrate_per_bed_list).transpose()
    #==========================================================================================================================
    Oil_Flowrate_list = []
    for n in range(len(permeability_array)):
        Oil_Flowrate = Oil_Flowrate_per_bed_table[n].sum()
        Oil_Flowrate_list.append(Oil_Flowrate)
    Oil_Flowrate_table = pd.DataFrame(Oil_Flowrate_list).rename(columns={0:'Oil Production Rate'})
    #===========================================================================================================================
    # CALCULATING THE VERTICAL COVERAGE
    coverage_list = []
    Total_Number_of_layers = len(permeability_array)-1
    for number_layer_breakthrough in range(len(permeability_array)):
        coverage_individual = (number_layer_breakthrough+((Total_Number_of_layers-number_layer_breakthrough)*Mobility_Ratio/(Mobility_Ratio-1))-(1/(Mobility_Ratio-1))*np.sqrt(Mobility_Ratio**2+List_of_permeability_ratio_DataTable[number_layer_breakthrough][1:]*(1-Mobility_Ratio**2)).sum())/Total_Number_of_layers
        coverage_list.append(coverage_individual)
    #Table of vertical coverage of the reservoir when a given layer just broke through.
    coverage_table = pd.DataFrame(coverage_list, columns=['Vertical Coverage (Fraction)'])
    #============================================================================================================================
    WOR_denominator_ratio_list = []
    for denominator_index in range(len(permeability_array)):
        WOR_denominator_ratio = permeability_array[denominator_index]/np.sqrt(Mobility_Ratio**2+List_of_permeability_ratio_DataTable[denominator_index]*(1-Mobility_Ratio**2))
        WOR_denominator_ratio_list.append(WOR_denominator_ratio)
    WOR_denominator_ratio_table = pd.DataFrame(WOR_denominator_ratio_list)
        #WOR_denominator_ratio_table
    WOR_list = []
    for n in range(len(permeability_array)):
        # CALCULATING THE WATER OIL RATIO, WORn and generate table
        sum_of_permeability = bed_data_sort.PERMEABILITY.iloc[0:n].sum()
        #for number_layer_breakthrough in range(len(permeability_array)):
        WOR = sum_of_permeability/(WOR_denominator_ratio_table[n].sum())
        WOR_list.append(WOR)
    WOR_table = pd.DataFrame(WOR_list).rename(columns={0:'Water-Oil Ratio'})#,columns=['WATER-OIL RATIO'])
    #==========================================================================================================================
    #CALCULATING THE CUMULATIVE OIL RECOVERY AS EACH BED BREAKSTHROUGH.
    Cumulative_oil_recovery = (7758*Areal_sweep_efficiency_at_breakthrough*Gross_rock_volume_acre_ft*Average_porosity*(SOI-SOR)*coverage_table).rename(columns={'Vertical Coverage (Fraction)':'Cumulative Oil Recovery (Barrels)'})
    #============================================================================================================================
    #CALCULATING THE VOLUME OF WATER REQUIRED TO FILL-UP THE GAS SPACE.
    Water_volume_to_fillup_gas_space = 7758*Area_acres*bed_data_sort.THICKNESS*bed_data_sort.POROSITY*(SGI-Residual_gas_saturation)
    Water_volume_to_fillup_gas_space_table=pd.DataFrame(Water_volume_to_fillup_gas_space, columns = ['Water Volume For Gas Space Fill-Up'])
    #============================================================================================================================
    #CALCULATING THE PRODUCING WATER-OIL RATIO
    Producing_water_oil_ratio = (WOR_table*OFVF).rename(columns={'Water-Oil Ratio':'Producing Water-Oil Ratio'})
    Producing_water_oil_ratio
    #===========================================================================================================================
    # Note that the integration for the calculation of the cumulative oil produced starts from 0
    # Hence, a new row will have to be inserted at the first row with element 0
    # this is done for both the producing water-oil ratio and the cumulative oil produced.

    # for the cumulative oil recovery
    Cumulative_oil_recovery.loc[-1] = [0]  # adding a row
    #Cumulative_oil_recovery.index = Cumulative_oil_recovery.index + 1  # shifting index
    Cumulative_oil_recovery_Starting_from_0 = Cumulative_oil_recovery.sort_index()  # sorting by index

    # for the producing water-oil ratio
    Producing_water_oil_ratio.loc[-1] = [0]  # adding a row
    #Producing_water_oil_ratio.index = Producing_water_oil_ratio.index + 1  # shifting index
    Producing_water_oil_ratio_Starting_from_0 = Producing_water_oil_ratio.sort_index()  # sorting by index

    # CALCULATING THE CUMULATIVE WATER PRODUCTION
    # To determine the cumulative water production, the produced water oil ratio is ingreated against the cumulative oil recovery.
    # The integration uses a cumulative trapezoidal row by row integration.
    # import numpy and scipy.integrate.cumtrapz 
    import numpy as np 
    from scipy import integrate
    # Preparing the Integration variables y, x.
    # the to.numpy() method converts from dataframe to numpy array which appears in the form of list of lists in the array. 
    #The concatenate function helps to bring the list of lists together.
    x = np.concatenate(Cumulative_oil_recovery_Starting_from_0.to_numpy(),axis=0)
    y = np.concatenate(Producing_water_oil_ratio_Starting_from_0.to_numpy(),axis=0) 
    # using scipy.integrate.cumtrapz() method 
    Cumulative_water_produced = pd.DataFrame(integrate.cumtrapz(y, x), columns = ['Cumulative Water Produced'])
    #==============================================================================================================================
    # CALCULATING THE CUMULATIVE WATER INJECTED, Wi
    Cumulative_water_injected = (Cumulative_water_produced['Cumulative Water Produced'] + OFVF*Cumulative_oil_recovery['Cumulative Oil Recovery (Barrels)'] + Water_volume_to_fillup_gas_space_table['Water Volume For Gas Space Fill-Up']).drop([-1])
    Cumulative_water_injected_table = pd.DataFrame(Cumulative_water_injected,columns = ['Cumulative Water Injected (Barrels)'])
    #===================================================================================================================================
    # CALCULATING THE TIME REQUIRED FOR INJECTION TO REACH A GIVEN RECOVERY.
    Time_days = Cumulative_water_injected_table['Cumulative Water Injected (Barrels)']/Constant_injection_rate
    Time_days_table = pd.DataFrame(Time_days).rename(columns ={'Cumulative Water Injected (Barrels)': 'Time (Days)'}, inplace = False)
    #print(Time_days_table)
    Time_years = Time_days_table/365
    Time_years_table = Time_years.rename(columns ={'Time (Days)': 'Time (Years)'}, inplace = False)
    #=======================================================================================================================================
    # TABLE OF ALL OBTAINED VALUES.
    All_tables =pd.concat([Layer_table,Oil_Mobility_table,Water_Mobility_table,
                           Water_Flowrate_per_bed_table, coverage_table, WOR_table,
                           Cumulative_oil_recovery, Water_volume_to_fillup_gas_space_table,
                           Producing_water_oil_ratio, Cumulative_water_produced,
                           Cumulative_water_injected_table, Time_days_table,
                           Time_years_table,Water_Flowrate_table,Oil_Flowrate_table,
                           Oil_Flowrate_per_bed_table.rename(columns=lambda x: str(x)+'Oil_Flowrate'),
                           Front_Location_list_DataTable.rename(columns=lambda x: str(x)+'Front_Location')
                           ], axis = 1).drop([-1])

       
    def results_and_graph_gui():
        global All_tables
        class MyTable(Table):

            def __init__(self, parent=None, **kwargs):
                Table.__init__(self, parent, **kwargs)
                return

        class MyApp1(Frame):

            def __init__(self, parent=None):
                self.parent = parent
                Frame.__init__(self)
                #self.main = self.master
                self.D_window = self.master
                #self.main.geometry('800x600+200+100')
                self.D_window.title('Dykstra-Parson General data and graph')
                f = Frame(self.D_window)
                f.pack(side = LEFT, expand=1)
                pt = make_table(f)
                bp = Frame(self.D_window)
                bp.pack(side=TOP)

                return

        def make_table(table_frame2, **kwds):

            df = All_tables
            pt = MyTable(table_frame2, dataframe=df, **kwds )
            pt.show()
            return pt
        def test1():
            """just make a table"""

            t = Toplevel()
            fr = Frame(t)
            fr.pack(fill=BOTH, expand=1)
            pt = make_table(fr)
            return

        def select_test():
            """cell selection and coloring"""

            #t = Toplevel()
            #fr = Frame(t)
            fr = table_frame2
            #fr.pack(fill=BOTH, expand=1)
            pt = Table(fr)
            pt.show()
            pt.All_tables
            pt.resetIndex(ask=False)
            pt.columncolors = {'c':'#337ab7'}
            df = pt.model.df

            mask_1 = df.a<7
            pt.setColorByMask('a', mask_1, '#337ab7')
            colors = {'red':'#f34130','blue':'blue'}
            for l in df.label.unique():
                mask = df['label']==l
                pt.setColorByMask('label', mask, l) 
            pt.redraw()
            return

        def multiple_tables():
            """make many tables in one frame"""

            t = Toplevel(height=800)
            r=0;c=0
            for i in range(6):
                fr = Frame(t)
                fr.grid(row=r,column=c)
                pt = make_table(fr, showtoolbar=True, showstatusbar=True)
                c+=1
                if c>2:
                    c=0
                    r+=1
            return
        app1 = MyApp1()
        app1.mainloop()
    #===========================================================================================================================
    table_frame2 = Frame(D_window)
    table_frame2.place(relheight = 1, relwidth =1)
    Button(table_frame2, text = 'View data',bg = '#337ab7',fg = 'white',justify = LEFT,relief= RAISED,cursor='hand2',command = results_and_graph_gui).grid(row=2,column =2,padx=5,pady=5,stick = W)    
    
    #================================================================================================================================
    D_window.geometry("200x150")
    D_window.mainloop() 


menubar = Menu(root)
#Create a load menu
loadmenu = Menu(menubar, tearoff=0)
loadmenu.add_command(label="Load Data",command =  Load_File)
menubar.add_cascade(label="Load Data", menu=loadmenu)

#newmenu = Menu(menubar, tearoff=0)
#newmenu.add_command(label="New",command = new())
#menubar.add_cascade(label="New", menu=newmenu)

Fractional_flowmenu = Menu(menubar, tearoff=0)
Fractional_flowmenu.add_command(label="Data and Plot",command =  fractional_flow)
menubar.add_cascade(label="Fractional Flow", menu=Fractional_flowmenu)

printmenu = Menu(menubar, tearoff=0)
printmenu.add_command(label="Print Result",command = (lambda: Resultfile(entries)))
menubar.add_cascade(label="Print Result", menu=printmenu)

# create the Output menu
#output = Menu(menu)
outputmenu = Menu(menubar, tearoff=0)
outputmenu.add_command(label="Tabular results and graphs",command =(lambda: D_tables(entries)))
#outputmenu.add_command(label="Fractional Flow", command = fractional_flow)
#added "file" to our menu
menubar.add_cascade(label="Dykstra-Parson", menu=outputmenu)

reznikmenu = Menu(menubar, tearoff=0)
reznikmenu.add_command(label="Reznik et al continuous solution", command=(lambda: Reznik(entries)))
#helpmenu.add_command(label="About...", command=donothing)
menubar.add_cascade(label="Reznik et al", menu=reznikmenu)
menubar.add_cascade(label="                                 Note : All Expressions Are In FIELD UNITS")

root.config(menu=menubar)
#************************************************************************************************************************
def Resultfile(entries):
    Number_of_points = float(entries['Number_of_points'].get())
    Length_of_bed_ft = float(entries['Length_of_bed_ft'].get())
    width_of_bed_ft = float(entries['width_of_bed_ft'].get())
    SWI = float(entries['SWI'].get())
    SOR = float(entries['SOR'].get())
    SOI = float(entries['SOI'].get())
    OFVF = float(entries['OFVF'].get())
    WFVF = float(entries['WFVF'].get())
    SGI = float(entries['SGI'].get())
    VISW = float(entries['VISW'].get())
    VISO = float(entries['VISO'].get())
    Saturation_gradient = float(entries['Saturation_gradient'].get())
    Constant_injection_rate = float(entries['Constant_injection_rate'].get())
    Inj_Pressure_differential = float(entries['Inj_Pressure_differential'].get())
    RGSU = float(entries['Residual_gas_saturation_unswept_area'].get())
    RGSS =  float(entries['Residual_gas_saturation_swept_area'].get())
    RGS = RGSU+RGSS
    import numpy as np
    KRW_1_SOR = np.interp(1-SOR, SW, KRW)
    KRO_SWI = np.interp(SWI, SW, KRO)
    Oil_Mobility =  permeability_array*KRO_SWI/VISO
    
    Water_Mobility =  permeability_array*KRW_1_SOR/VISW
    
    Mobility_Ratio = KRW_1_SOR*VISO/(KRO_SWI*VISW)
    Areal_sweep_efficiency_at_breakthrough = 0.54602036+(0.03170817/Mobility_Ratio)+(0.30222997/math.exp(Mobility_Ratio)-0.0050969*Mobility_Ratio)
    Displacement_efficiency = (1-SWI-SGI-SOR)/(1-SWI-SGI)
    Areal_sweep_efficiency =  Areal_sweep_efficiency_at_breakthrough+0.2749*np.log((1/Displacement_efficiency))
    
    #Average_porosity = np.mean(bed_data_sort.POROSITY)
    Average_porosity = np.mean(bed_data_sort.POROSITY)
    Area_acres = Length_of_bed_ft*width_of_bed_ft/43560
    Gross_rock_volume_acre_ft =  Area_acres*bed_data_sort.THICKNESS.sum()

    SW_table = pd.DataFrame(SW, columns = ['SW'])
    b = (np.log((KRO/KRW)[2])-np.log((KRO/KRW)[3]))/(SW[3]-SW[2])
    a = (KRO/KRW)[2]*math.exp(b*SW[2])
    def fw(SW):
        fw = 1/(1+a*(VISW/VISO)*np.exp(-b*SW))
        return(fw)
    xList = []
    for i in range(0, 10000):
        x = random.uniform(SWI+0.1, 1)
        xList.append(x) 
    xs = np.array(xList)
    m = 1/((xs-SWI)*(1+(VISW/VISO)*a*np.exp(-b*xs)))
    tangent_slope=max(m)
    Saturation_at_Breakthrough = SWI + 1/tangent_slope
    def funct(SWF):
        swf = SWF[0]
        F = np.empty((1))
        F[0] = ((tangent_slope*(swf-SWI)*(1+(VISW/VISO)*a*math.exp(-b*swf)))-1)
        return F
    SWF_Guess = np.array([SWI+0.1])
    SWF = fsolve(funct, SWF_Guess)[0]
    Fwf = fw(SWF)
    Fw = fw(SW)
    Fw_table = pd.DataFrame(Fw, columns = ['Fractional Flow(Fw)'])
    dfw_dSw = (VISW/VISO)*a*b*np.exp(-SW*b)/(1+(VISW/VISO)*a*np.exp(-SW*b))**2
    dfw_dSw_table = pd.DataFrame(dfw_dSw, columns = ['dFw/dSw'])

    tangent = (SW-SWI)*tangent_slope
    tangent_table = pd.DataFrame(tangent, columns = ['Tangent'])
    Fractional_flow_table = pd.concat([SW_table, Fw_table, dfw_dSw_table, tangent_table], axis=1)
        
    f = open('Result.txt', 'w')
    f.write('RUN DATE AND TIME: '+ str(date_time)+'\n')
    f.write('\n')
    f.write('INPUTS \n')
    f.write('Number of points: '+ str(Number_of_points)+'\n')
    f.write('Lenght of bed in feet: '+ str(Length_of_bed_ft)+'\n')
    f.write('Width of bed in feet: '+ str(width_of_bed_ft)+'\n')
    f.write('Average porosity: '+ str(average_porosity)+'\n')
    f.write('Viscosity of oil in centipoise: '+ str(VISO)+'\n')
    f.write('Viscosity of water in centipoise: '+ str(VISW)+'\n')
    f.write('Formation volume factor of oil: '+ str(OFVF)+'\n')
    f.write('Formation volume factor of water: '+ str(WFVF)+'\n')
    f.write('Initial water saturation: '+ str(SWI)+'\n')
    f.write('Initial gas saturation: '+ str(SGI)+'\n')
    f.write('Initial oil saturation: '+ str(SOI)+'\n')
    f.write('Residual oil saturation: '+ str(SOR)+'\n')
    f.write('Constant injection rate (bbl/day): '+ str(Constant_injection_rate)+'\n')
    f.write('Injection pressure (psi): '+ str(Inj_Pressure_differential)+'\n')
    f.write('Residual gas saturation of unswept area: '+ str(RGSU)+'\n')
    f.write('Residual gas saturation of swept area: '+ str(RGSS)+'\n')
    f.write('Residual gas saturation: '+ str(RGS)+'\n')
    f.write('\n')
    f.write('***************************************************************************************************************\n ')
    f.write('\n')
    f.write('GENERAL OUTPUTS \n')
    f.write('Average porosity: '+ str(Average_porosity)+'\n')
    f.write('Water relative permeability at 1-SOR: '+ str(KRW_1_SOR)+'\n')
    f.write('Oil relative permeability at initial water saturation: '+ str(KRO_SWI)+'\n')
    f.write('Mobility ratio: '+ str(Mobility_Ratio)+'\n')
    f.write('Areal sweep efficiency at breakthrough: '+ str(Areal_sweep_efficiency_at_breakthrough)+'\n')
    f.write('Area of reservoir bed in acres: '+ str(Area_acres)+'\n')
    f.write('Gross rock volume in acres-feet: '+ str(Gross_rock_volume_acre_ft)+'\n')
    f.write('Displacement efficiency: '+ str(Displacement_efficiency)+'\n')
    f.write('Areal sweep efficiency: '+ str(Areal_sweep_efficiency)+'\n')
    f.write('Saturation gradient: '+ str(Saturation_gradient)+'\n')
    f.write('\n')
    f.write('FRACTIONAL FLOW OUTPUTS \n')
    f.write('Correlation: Kro/Krw = aexp(-bSw) \n')
    f.write('b: '+ str(b)+'\n')
    f.write('a: '+ str(a)+'\n')
    f.write('Slope of the tangent line: '+ str(tangent_slope)+'\n')
    f.write('Flood Front Saturation(Swf): '+ str(SWF)+'\n')
    f.write('Flood Front Fractional Flow(Fwf): '+ str(Fwf)+'\n')
    f.write('Saturation at breakthrough(SwBT): '+ str(Saturation_at_Breakthrough)+'\n')
    f.write('\n')
    f.write(str(Fractional_flow_table)+'\n')
    f.write('\n')
    f.write('DYKSTRA-PARSONS OUTPUTS \n')
    f.write(str(All_tables)+'\n')
    f.write('\n')
    f.write('REZNIK ET AL OUTPUTS \n')
    f.write(str(General)+'\n')
    f.close()
    return f
#====================================================================================================================================================
root.geometry("900x500")
root.mainloop() 
