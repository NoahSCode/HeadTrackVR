import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import app

def show_popup(output_location):
    message = f"Video analysis done. Excel file 'movement_log.xlsx' saved to {output_location}"
    messagebox.showinfo("Analysis Complete", message)

def analyze():
    video_file = video_file_entry.get()
    output_file = output_file_entry.get()
    threshold = int(threshold_entry.get())
    app.process_video(video_file, output_file, threshold)
    # Show the popup window after the analysis is done
    show_popup(output_file)
# Function to browse files
def browse_file(entry_widget):
    filename = filedialog.askopenfilename()
    entry_widget.delete(0, 'end')
    entry_widget.insert(0, filename)

# Function to browse directory
def browse_directory(entry_widget):
    directory = filedialog.askdirectory()
    entry_widget.delete(0, 'end')
    entry_widget.insert(0, directory)

# Create the main window
root = tk.Tk()
root.title("GUI Wrapper for HeadTrackVR")
root.configure(bg="#F0F0F0")

# Create a frame for the description
desc_frame = tk.Frame(root, pady=10, bg="#F0F0F0")
desc_frame.pack()

# Create a Label for the first description
desc1_label = tk.Label(desc_frame, text="GUI Wrapper for HeadTrackVR", font=("Helvetica", 16, "bold"), bg="#F0F0F0")
desc1_label.pack()

# Create a Label for the second description
desc2_label = tk.Label(desc_frame, text="Tracks head movement in a screen recording of a VR environment and calculates the amount of time participant spends moving their head in the virtual space.", wraplength=500, bg="#F0F0F0")
desc2_label.pack()

# Create a frame for the input fields
input_frame = tk.Frame(root, padx=20, pady=20, bg="#F0F0F0")
input_frame.pack()

# Create a Label and Spinbox for the threshold. Default input value is set to 10000000.
threshold_label = tk.Label(input_frame, text="Threshold:", font=("Helvetica", 12), bg="#F0F0F0")
threshold_label.grid(row=0, column=0, sticky="w")
threshold_value = tk.StringVar()
threshold_value.set("10000000")
threshold_entry = tk.Entry(input_frame, textvariable=threshold_value, font=("Helvetica", 12), width=10)
threshold_entry.grid(row=0, column=1, sticky="w")



# Create a Label, Entry, and Button for the video file location
video_file_label = tk.Label(input_frame, text="Video file location:", font=("Helvetica", 12), bg="#F0F0F0")
video_file_label.grid(row=1, column=0, sticky="w")
video_file_entry = tk.Entry(input_frame, font=("Helvetica", 12))
video_file_entry.grid(row=1, column=1, sticky="w")
video_file_button = tk.Button(input_frame, text="Browse", command=lambda: browse_file(video_file_entry))
video_file_button.grid(row=1, column=2, padx=10)

# Create a Label, Entry, and Button for the output excel file
output_file_label = tk.Label(input_frame, text="Output excel file:", font=("Helvetica", 12), bg="#F0F0F0")
output_file_label.grid(row=2, column=0, sticky="w")
output_file_entry = tk.Entry(input_frame, font=("Helvetica", 12))
output_file_entry.grid(row=2, column=1, sticky="w")
output_file_button = tk.Button(input_frame, text="Browse", command=lambda: browse_directory(output_file_entry))
output_file_button.grid(row=2, column=2, padx=10)

# Create a frame for the analyze button
analyze_frame = tk.Frame(root, padx=20, bg="#F0F0F0")
analyze_frame.pack()

# Create a button for executing the analysis
analyze_button = tk.Button(analyze_frame, text="Analyze", font=("Helvetica", 12), command=analyze)
analyze_button.pack()

# Run the tkinter event loop
root.mainloop()