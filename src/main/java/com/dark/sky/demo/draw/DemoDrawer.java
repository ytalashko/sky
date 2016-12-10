package com.dark.sky.demo.draw;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

public class DemoDrawer {

  public DemoDrawer(Function<List<Double>, Integer> predictor, Supplier<List<Double>> dataSupplier) {
    JFrame frame = new JFrame("Test examples");
    frame.setBounds(600, 50, 200, 270);
    frame.setLayout(null);
    frame.setResizable(false);

    List<List<Double>> dataContainer = new ArrayList<>(1);
    dataContainer.add(0, dataSupplier.get());
    int prediction = predictor.apply(dataContainer.get(0));

    ImageDrawer imageDrawer = new ImageDrawer(() -> dataContainer.get(0));
    imageDrawer.setSize(ImageDrawer.SIZE, ImageDrawer.SIZE);
    imageDrawer.setLocation(0, 0);
    frame.add(imageDrawer);

    JLabel label = new JLabel("Prediction: " + prediction);
    label.setSize(120, 10);
    label.setLocation(55, 210);
    frame.add(label);

    JButton button = new JButton("Next");
    button.setSize(120, 30);
    button.setLocation(40, 230);
    button.addActionListener(new AbstractAction() {
      @Override
      public void actionPerformed(ActionEvent e) {
        dataContainer.set(0, dataSupplier.get());
        int prediction = predictor.apply(dataContainer.get(0));
        label.setText("Prediction: " + prediction);
        frame.repaint();
      }
    });
    frame.add(button);

    frame.setVisible(true);
    frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
  }
}
