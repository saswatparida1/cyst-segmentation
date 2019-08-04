from gi.repository import Gtk, Gio
from gi.repository.Gtk import FileChooserDialog, Box
import convert
import subprocess
import shutil
from PIL import Image


class MainWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="MEDICINE REQUIREMENT CALCULATION")
        self.set_border_width(10)
        self.set_default_size(1000, 500)
        self.set_icon_from_file("eye.jpg")
        self.spinner = Gtk.Spinner()
        

        layout = Gtk.Box( spacing=8)  # type: Box
        self.add(layout)
        #sideview
        self.liststore = Gtk.ListStore(str, str)
        self.liststore.append(["Images Type", "OCT"])
        self.liststore.append(["Training Model", "UNET-CNN"])
        self.liststore.append(["Training Pixel", "256*512"])
        self.liststore.append(["Error Function", "binary_crossentropy"])
        self.liststore.append(["Optimizer", "adam"])

        treeview = Gtk.TreeView(model=self.liststore)
        renderer_text = Gtk.CellRendererText()
        column_text = Gtk.TreeViewColumn("INFO", renderer_text, text=0)
        treeview.append_column(column_text)
        renderer_editabletext = Gtk.CellRendererText()
        column_editabletext = Gtk.TreeViewColumn("Value",
                                                 renderer_editabletext, text=1)
        treeview.append_column(column_editabletext)


        # titlebar
        header_bar = Gtk.HeaderBar(name='lay')
        header_bar.set_show_close_button(True)
        header_bar.props.title = "CYST SEGMENTATION"
        self.set_titlebar(header_bar)
        main_menu = Gtk.Toolbar()

        file_new = Gtk.ToolButton(Gtk.STOCK_NEW)
        file_new1 = Gtk.ToolButton(Gtk.STOCK_OPEN)
        file_new3 = Gtk.ToolButton(Gtk.STOCK_QUIT)
        
        file_new4 = Gtk.ToolButton(Gtk.STOCK_SAVE)
        file_new.connect("clicked", self.on_new)

        self.vbox1=Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        self.im = Gtk.Image(name='i1')
        self.im.set_size_request(525,260)
        self.im1 = Gtk.Image(name='i2')
        self.im1.set_size_request(525,260)
        self.im2 = Gtk.Image(name='i3')
        self.im2.set_size_request(525,260)

        file_new1.connect("clicked", self.on_file_clicked)
        self.vbox1.add(treeview)
        if(self.im):
            self.vbox1.add(self.im)
        
        


        file_new3.connect("clicked", self.closing)
        file_new4.connect("clicked", self.saving)

        main_menu.insert(file_new,0)
        main_menu.insert(file_new1,1)
        main_menu.insert(file_new4,2)
        main_menu.insert(file_new3,3)

        #main_menu.append(file_menu_dropdown)
       # header_bar.add(button)
        header_bar.pack_end(main_menu)
        # input
        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        self.add(self.vbox)


        self.button = Gtk.Button(label="PROCESS",name='process')
        self.button.connect("clicked", self.connection)
        self.vbox1.add(self.button)
        self.vbox1.pack_start(self.spinner, True, True, 0)
        layout.add(self.vbox1)

        self.add(Gtk.TextView())
        # images



        if (self.im1):
            self.vbox.add(self.im1)
        if (self.im2):
            self.vbox.add(self.im2)
        layout.add(self.vbox)
    def saving(self,widget):
        dialog1 = Gtk.FileChooserDialog("Save your PREDICTED image", self,
                                       Gtk.FileChooserAction.SAVE,
                                       (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                        Gtk.STOCK_SAVE, Gtk.ResponseType.ACCEPT))
        dialog2 = Gtk.FileChooserDialog("Save your OVERLAY image", self,
                                       Gtk.FileChooserAction.SAVE,
                                       (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                        Gtk.STOCK_SAVE, Gtk.ResponseType.ACCEPT))
        self.add(dialog1)
        
        self.add(dialog2)

        Gtk.FileChooser.set_do_overwrite_confirmation(dialog1, True)
        Gtk.FileChooser.set_do_overwrite_confirmation(dialog2, True)
    
        response = dialog1.run()

        if response == Gtk.ResponseType.ACCEPT:
            filename= Gtk.FileChooser.get_filename(dialog1)
            shutil.copy2("/home/saswat/PycharmProjects/saswat/name1.png",filename)
        
        dialog1.destroy()
            
        response=dialog2.run() 
        
        if response == Gtk.ResponseType.ACCEPT:
            filename= Gtk.FileChooser.get_filename(dialog2)           
            shutil.copy2("/home/saswat/PycharmProjects/saswat/overlay.png",filename)

        dialog2.destroy()

    def on_new(self, widget):
        self.im.clear()
        self.im1.clear()
        self.im2.clear()


    def on_file_clicked(self, widget):

        if(self.im1):
            self.im1.clear()

        if (self.im2):
            self.im2.clear()
        dialog = Gtk.FileChooserDialog("Please choose a file", self,
                                       Gtk.FileChooserAction.OPEN,
                                       (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                        Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
        self.add(dialog)
        
        response=dialog.run()
        if response == Gtk.ResponseType.OK:
            img = Image.open(dialog.get_filename()) 
            new_width  = 512
            new_height = 256
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
            img.save('/home/saswat/PycharmProjects/saswat/input.png')
            
            self.im.set_from_file('/home/saswat/PycharmProjects/saswat/input.png')
            self.ime = convert.conv('/home/saswat/PycharmProjects/saswat/input.png')


        dialog.destroy()




    def connection(self, widget):
        self.spinner.start()
        
        if(self.im):

            import numpy as np
            ime1=np.asarray(self.ime)
            print(ime1.shape)
            np.save('sas.npy',ime1)

            subprocess.run(['python3','ui.py'])

            self.spinner.stop()
            self.im1.set_from_file("/home/saswat/PycharmProjects/saswat/name1.png")

            subprocess.run(['python3','ex.py'])
            self.im2.set_from_file("/home/saswat/PycharmProjects/saswat/overlay.png")



    def closing(self, widget):
        quit(0)

window = MainWindow()
window.connect("delete-event", Gtk.main_quit)
window.show_all()
Gtk.main()

