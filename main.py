"""Entry point for T-Shirt Helper."""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from app.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("T-Shirt Helper")

    window = MainWindow()
    window.setAcceptDrops(True)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
