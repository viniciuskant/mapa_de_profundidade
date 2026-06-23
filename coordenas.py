#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class PointSelector:
    def __init__(self, image_path, n_points):
        self.n_points = n_points
        self.points = []
        self.artists = []

        self.img = mpimg.imread(image_path)

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img)
        self.ax.set_title(
            f"Selecione {n_points} pontos\n"
            "Clique para adicionar | Ctrl+Z ou 'u' para desfazer"
        )

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def draw_point(self, x, y, idx):
        marker, = self.ax.plot(
            x, y,
            marker="o",
            color="red",
            markersize=6
        )

        label = self.ax.text(
            x + 5,
            y + 5,
            str(idx),
            color="yellow",
            fontsize=12,
            weight="bold",
            bbox=dict(facecolor="black", alpha=0.5, pad=1)
        )

        self.artists.append((marker, label))

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        if len(self.points) >= self.n_points:
            return

        x, y = event.xdata, event.ydata

        idx = len(self.points)
        self.points.append((x, y))
        self.draw_point(x, y, idx)

        self.fig.canvas.draw_idle()

        print(f"P{idx}: ({x:.2f}, {y:.2f})")

        if len(self.points) == self.n_points:
            print("\nNúmero de pontos atingido.")
            plt.close(self.fig)

    def undo(self):
        if not self.points:
            print("Nenhum ponto para desfazer.")
            return

        self.points.pop()

        marker, label = self.artists.pop()
        marker.remove()
        label.remove()

        self.fig.canvas.draw_idle()

        print(f"Removido ponto {len(self.points)}")

    def on_key(self, event):
        key = event.key.lower() if event.key else ""

        if key in ["u", "ctrl+z"]:
            self.undo()

    def run(self):
        plt.show()

        print("\n=== Coordenadas finais ===")
        for i, (x, y) in enumerate(self.points):
            print(f"[{x:.0f}, {y:.0f}],")


def main():
    if len(sys.argv) != 3:
        print(f"Uso: {sys.argv[0]} <imagem> <num_pontos>")
        sys.exit(1)

    image_path = sys.argv[1]
    n_points = int(sys.argv[2])

    selector = PointSelector(image_path, n_points)
    selector.run()


if __name__ == "__main__":
    main()