# UI for handwriting recognition

"""Modules to build a ui"""
import tkinter
import customtkinter

# The following section sets the defaults for the window
customtkinter.set_appearance_mode("system")
customtkinter.set_default_color_theme("dark-blue")

app = customtkinter.CTk()
app.title("Handwriting Recognition Tool")
app.geometry("750x500")
# app.grid_columnconfigure(0, weight=1)
app.grid_rowconfigure(0, weight=1)


# Labels:

# Text boxes: Will be to the bottoma as may depend on frames etc
IN_TEXT = """
    Input file path here
"""
OUT_TEXT = """
    Output file path here
"""
text_in_folder = customtkinter.CTkEntry(
    app, width=200, height=20, placeholder_text=IN_TEXT)
text_out_folder = customtkinter.CTkEntry(
    app, width=200, height=20, placeholder_text=OUT_TEXT)


# text_in_folder.insert("0.0", text=IN_TEXT)
# text_out_folder.insert("0.0", text=OUT_TEXT)
# text_in_folder.configure(state="disabled")
text_out_folder.configure(state="disabled")

# Buttons


def button_callback():
    print(text_in_folder.get())


button = customtkinter.CTkButton(
    app, text="Run", command=button_callback)


# Place UI elements
text_in_folder.grid(row=1, column=0, padx=10, pady=10)
text_out_folder.grid(row=1, column=2, padx=10, pady=10)
button.grid(row=2, column=1, padx=20, pady=20)


app.mainloop()
