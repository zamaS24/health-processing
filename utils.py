
# clears a layout from all it's children
def clear_layout(layout):
    while layout.count():
        widget = layout.itemAt(0).widget()  
        if widget is not None:
            widget.deleteLater() 
        layout.removeItem(layout.itemAt(0)) 