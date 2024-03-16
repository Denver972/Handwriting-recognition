# UI for handwriting recognition

"""Modules to build a ui"""
# import tkinter
import customtkinter

# The following section sets the defaults for the window
customtkinter.set_appearance_mode("light")
customtkinter.set_default_color_theme("dark-blue")

app = customtkinter.CTk()
app.title("Handwriting Recognition Tool")
app.geometry("750x400")
app.grid_columnconfigure(0, weight=1)
app.grid_columnconfigure(2, weight=1)
app.grid_rowconfigure(0, weight=1)


# Frames:
# Progress bar frame
prog_frame = customtkinter.CTkFrame(
    master=app, width=250, height=50, fg_color="transparent")
in_frame = customtkinter.CTkFrame(
    master=app, width=250, height=50, fg_color="transparent")
out_frame = customtkinter.CTkFrame(
    master=app, width=250, height=50, fg_color="transparent")

# Text boxes: Will be to the bottoma as may depend on frames etc
IN_TEXT = """
    Input file path here
"""
OUT_TEXT = """
    Output file path here
"""
text_in_folder = customtkinter.CTkEntry(
    in_frame, width=250, height=20, placeholder_text=IN_TEXT)
text_out_folder = customtkinter.CTkEntry(
    out_frame, width=250, height=20, placeholder_text=OUT_TEXT)


# text_in_folder.insert("0.0", text=IN_TEXT)
# text_out_folder.insert("0.0", text=OUT_TEXT)
# text_in_folder.configure(state="disabled")
# text_out_folder.configure(state="disabled")

# Labels:
in_label = customtkinter.CTkLabel(
    in_frame, text="Input Folder:", fg_color="transparent")
out_label = customtkinter.CTkLabel(
    out_frame, text="Output Folder:", fg_color="transparent")
prog_label = customtkinter.CTkLabel(
    prog_frame, text="Progress", fg_color="transparent")

# Progress bar:
progressbar = customtkinter.CTkProgressBar(
    prog_frame, orientation="horizontal", progress_color="sea green")
progressbar.set(0.0)

# Buttons


def button_callback():
    """ text to get rid of error"""
    print(text_in_folder.get())
    progressbar.start()


button = customtkinter.CTkButton(
    app, text="Run", command=button_callback)


# Place UI elements
in_frame.grid(row=1, column=0, padx=10)
in_label.pack()
text_in_folder.pack()
out_frame.grid(row=1, column=2, padx=10)
out_label.pack()
text_out_folder.pack()
button.grid(row=3, column=1, padx=10, pady=0)
prog_frame.grid(row=3, column=2, pady=10)
prog_label.pack()
progressbar.pack()


app.mainloop()
