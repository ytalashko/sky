package com.dark.sky.demo.draw;

import javax.swing.*;
import java.awt.*;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

class ImageDrawer extends JPanel {
  final static int SIZE = 200;

  private final Supplier<List<Double>> dataSupplier;

  ImageDrawer(Supplier<List<Double>> dataSupplier) {
    this.dataSupplier = dataSupplier;
  }

  @Override
  public void paintComponent(Graphics g) {
    Graphics2D g2 = (Graphics2D) g;

    List<Double> data = dataSupplier.get();

    double min = Collections.min(data);
    double max = Collections.max(data) - min;

    int index = 0;
    for (double i : data) {
      int l = (int) (Math.abs((i - min) / max) * 255);
      g2.setColor(new Color(l, l, l));
      g2.fillRect(index / SIZE * 10, index % SIZE, 10, 10);
      index += 10;
    }
  }
}
