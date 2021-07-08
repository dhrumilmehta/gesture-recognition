
/**
 * A dataset for webcam controls which allows the user to add example Tensors
 * for particular labels. This object will concat them into two large xs and ys.
 */

class ControllerDataset {
  constructor() {
    this.labels = []
  }

  /**
   * Adds an example to the controller dataset.
   * @param {Tensor} example A tensor representing the example. It can be an
   *     image, an activation, or any other type of Tensor.
   * @param {number} label The label of the example. Should be a number.
   */
  addExample(example, label) {
    // One-hot encode the label.
    const y = label;

    if (this.xs == null) {
      // For the first example that gets added, keep example and y so that the
      // ControllerDataset owns the memory of the inputs. This makes sure that
      // if addExample() is called in a tf.tidy(), these Tensors will not get
      // disposed.
      this.xs = tf.keep(example);
      this.labels.push(labelsCaptured.indexOf(y));
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));
      this.labels.push(labelsCaptured.indexOf(y));

      oldX.dispose();
    }
  }
  addLabels(numClasses) {
    for (var i = 0; i < this.labels.length; i++) {
      if (this.ys == null) {
        this.ys = tf.keep(tf.tidy(
            () => {return tf.oneHot(
                tf.tensor1d([this.labels[i]]).toInt(), numClasses)}));
      } else {
        const y = tf.tidy(
            () => {return tf.oneHot(
                tf.tensor1d([this.labels[i]]).toInt(), numClasses)});
        const oldY = this.ys;
        this.ys = tf.keep(oldY.concat(y, 0));
        oldY.dispose();
        y.dispose();
      }
    }
  }
}
