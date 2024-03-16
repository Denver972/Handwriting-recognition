# User interface file to make the program easy to use without needing to go
# into the code. The aim is to have a space to specify the input folder, an
# output folder initially Also there my be toggles to specify if there is
# debugging or intermediatee steps saved and shown. Additionally a progress bar
# will be included

"""Modules to build a ui"""
import tkinter
import customtkinter

# The following section sets the defaults for the window
customtkinter.set_appearance_mode("system")
customtkinter.set_default_color_theme("dark-blue")

app = customtkinter.CTk()
app.title("Handwriting Recognition Tool")
app.geometry("750x500")

# this section creates tables
tabView = customtkinter.CTkTabview(app)
tabView.add("Settings")
tabView.add("Extra1")
tabView.add("Extra2")
tabView.set("Settings")

# # frame template
frame = customtkinter.CTkFrame(
    master=app, width=50, height=50, corner_radius=0, bg_color="green")
frame2 = frame = customtkinter.CTkFrame(
    master=app, width=50, height=50, corner_radius=0, bg_color="green")

# This section creates Text Boxes
# Making two text boxes to tike the input and output file
text_in_folder = customtkinter.CTkTextbox(app, width=200, height=10)
text_out_folder = customtkinter.CTkTextbox(app)

IN_TEXT = """
    Input file path here
"""
OUT_TEXT = """
    Output file path here
"""
text_in_folder.insert("0.0", text=IN_TEXT)
text_out_folder.insert("0.0", text=OUT_TEXT)
text_in_folder.configure(state="disabled")
text_out_folder.configure(state="disabled")


# # Button template


def button_func():
    """This code will excecute when the button is pressed"""
    print(text_in_folder.get("0.0", "end"))


def settings_button_func():
    """Save the settings before running the program"""
    print("Settings Saved")


button = customtkinter.CTkButton(
    master=frame, text="Start Digitising", command=button_func)

settings_button = customtkinter.CTkButton(
    master=tabView.tab("Settings"), text="Save Settings",
    command=settings_button_func)

# This section populates the buttons, tables and frames


tabView.pack(padx=0, pady=10)
frame.pack(padx=10, pady=10)
frame2.pack(padx=10, pady=10)
text_in_folder.pack(padx=10, pady=10)
button.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)
settings_button.place(relx=0.5, rely=0.9, anchor=tkinter.CENTER)

app.mainloop()
