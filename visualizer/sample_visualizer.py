from datetime import time
import matplotlib.pyplot as plt

class InteractiveSampleViewer:
    def __init__(self, image_list, roi_mask, time_series, annotation_idxs: [int], peek_idxs: [int],
                 valley_idxs: [int], out_breaths: int, in_breaths: int, annotated_rate: float, calculated_rate: float,
                 duration: time):
        """
        Initialize the interactive viewer.

        :param image_list: List of m*n*1 images (np.ndarray) to visualize.
        :param roi_mask: m*n ROI mask (np.ndarray) to overlay on the images.
        :param time_series: List of time series.
        :param annotation_idxs: Indices for annotated events.
        :param peek_idxs: Indices for peeks in the time series.
        :param valley_idxs: Indices for valleys in the time series.
        :param out_breaths: Number of out breaths.
        :param in_breaths: Number of in breaths.
        :param annotated_rate: Annotated breathing rate.
        :param calculated_rate: Calculated breathing rate.
        :param duration: Total duration of the video.
        """
        self.valley_idxs = valley_idxs
        self.peek_idxs = peek_idxs
        self.image_list = [img.squeeze() for img in image_list]  # Ensure images are 2D
        self.roi_mask = roi_mask
        self.time_series = time_series
        self.current_index = 0
        self.out_breaths = out_breaths
        self.int_breaths = in_breaths
        self.annotated_rate = annotated_rate
        self.calculated_rate = calculated_rate
        self.duration = duration

        self.fig, (self.ax_top, self.ax_ts) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 3]})
        self.ax_img = self.ax_top  # Image and text will go in the top axis
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.im_display = self.ax_img.imshow(self.image_list[self.current_index], cmap="gray")
        self.mask_display = self.ax_img.imshow(
            self.roi_mask, cmap="Reds", alpha=0.3, interpolation="none"
        )
        self.ax_img.set_title(f"Image 1 of {len(self.image_list)}")

        self.text_box = self.ax_img.text(
            1.05, 0.5, self.get_text_info(), transform=self.ax_img.transAxes, fontsize=20,
            verticalalignment='center', horizontalalignment='left', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

        self.ax_ts.set_title("Time Series")
        self.ax_ts.set_xlabel("Time Index")
        self.ax_ts.set_ylabel("Value")
        for i, series in enumerate(self.time_series):
            if i == len(self.time_series) - 1:
                self.ax_ts.plot(series[0], color=series[2], label=series[1])
            else:
                self.ax_ts.plot(series[0], color=series[2], label=series[1], alpha=0.075)
        self.ax_ts.set_ylim(0, 1)
        self.ax_ts.legend()

        self.current_vline = None
        self.update_annotations(annotation_idxs)
        self.update_peeks_and_valleys()

        plt.tight_layout()
        plt.show()

    def get_text_info(self):
        """Generate the text information to display beside the image."""
        return (f"Annotated Rate: {self.annotated_rate:.2f}\n"
                f"Calculated Rate: {self.calculated_rate:.2f}\n"
                f"Duration: {self.duration}")

    def on_key_press(self, event):
        """
        Handle key press events for navigation.

        Left arrow: Show the previous image.
        Right arrow: Show the next image.
        """
        if event.key == "right":
            self.current_index = (self.current_index + 1) % len(self.image_list)
        elif event.key == "left":
            self.current_index = (self.current_index - 1) % len(self.image_list)

        self.update_display()

    def update_display(self):
        """Update the display with the current image, ROI mask, and time series."""
        # Update the image display
        self.im_display.set_data(self.image_list[self.current_index])
        self.mask_display.set_data(self.roi_mask)
        self.ax_img.set_title(f"Image {self.current_index + 1} of {len(self.image_list)}")

        if self.current_vline:
            self.current_vline.remove()

        self.current_vline = self.ax_ts.axvline(self.current_index, color='red', linestyle='--', label="Current Image")

        self.text_box.set_text(self.get_text_info())

        self.fig.canvas.draw_idle()

    def update_annotations(self, annotation_frames: [int]):
        """
        Draw vertical lines at each annotation frame on the time series plot.
        :param annotation_frames: List of frame indices where annotations are present.
        """
        self.ax_ts.axvline(annotation_frames[0], color='darkgreen', linestyle=':', label='Annotated Frame')
        for frame in annotation_frames[1:]:
            self.ax_ts.axvline(frame, color='darkgreen', linestyle=':')

        self.fig.canvas.draw_idle()

    def update_peeks_and_valleys(self):
        """
        Draw vertical lines at each peek and valley on the time series plot.
        """
        self.ax_ts.axvline(self.peek_idxs[0], color='darkblue', linestyle='--', label='Peeks')
        for peek in self.peek_idxs[1:]:
            self.ax_ts.axvline(peek, color='darkblue', linestyle='--')

        self.ax_ts.axvline(self.valley_idxs[0], color='darkred', linestyle='--', label='Valleys')
        for valley in self.valley_idxs[1:]:
            self.ax_ts.axvline(valley, color='darkred', linestyle='--')

        self.fig.canvas.draw_idle()
