import tkinter as tk

def main():
    position = []
    edges = []
    with open("param.csv") as fp:
        cnt = 0
        for line in fp:
            if cnt == 0:
                cnt = 1
                continue
            xy = line.split(",")
            position.append((float(xy[0]) + 1.1, float(xy[1]) + 1.1))
    
    with open("edge.csv") as fp:
        cnt = 0
        for line in fp:
            if cnt == 0:
                cnt = 1
                continue
            xy = line.split(",")
            edges.append((int(xy[0]), int(xy[1])))
    
    root = tk.Tk()
    cv = tk.Canvas(root, bg = "white", width = 1000, height = 800)
    for edge in edges:
        cv.create_line(position[edge[0]][0] * 300, position[edge[0]][1] * 300, position[edge[1]][0] * 300, position[edge[1]][1] * 300)
    cv.pack()
    root.mainloop()


if __name__ == "__main__":
    main()