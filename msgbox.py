import tkinter as tk

def msg(head,msg):
    root = tk.Tk()
    root.title(head)
    root.aspect=150

    #width = 350
    #frm_width = root.winfo_rootx() - root.winfo_x()
    #win_width = width + 2 * frm_width
    #height = 180
    #titlebar_height = root.winfo_rooty() - root.winfo_y()
    #win_height = height + titlebar_height + frm_width
    #x = root.winfo_screenwidth() // 2 - win_width // 2
    #y = root.winfo_screenheight() // 2 - win_height // 2
    root.geometry("")
    label = tk.Label(root, text=msg)
    label.config(anchor="center")
    label.pack(side="top", fill="both", expand=True, padx=10, pady=5)
    button = tk.Button(root, width=6, text="OK", command=lambda: root.destroy())
    button.pack(side="bottom", fill="none", expand=True, pady=5)
    root.mainloop()

    #    root = tk.Tk()
    #root.title(head)
    #root.aspect=150

    #width = 350
    #frm_width = root.winfo_rootx() - root.winfo_x()
    #win_width = width + 2 * frm_width
    #height = 180
    #titlebar_height = root.winfo_rooty() - root.winfo_y()
    #win_height = height + titlebar_height + frm_width
    #x = root.winfo_screenwidth() // 2 - win_width // 2
    #y = root.winfo_screenheight() // 2 - win_height // 2
    #root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    #label = tk.Label(root, text=msg)
    #label.config(anchor="center")
    #label.pack(side="top", fill="both", expand=True, padx=10, pady=5)
    #button = tk.Button(root, width=6, text="OK", command=lambda: root.destroy())
    #button.pack(side="bottom", fill="none", expand=True, pady=5)
    #root.mainloop()


