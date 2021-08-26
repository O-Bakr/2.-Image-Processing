import cv2
import numpy as np
import edge_detection as edge
import matplotlib.pyplot as plt

class Lane:
    """
    Represents a lane on a road.
    """
    def __init__(self, orig_frame):
        """
          Default constructor

        :param orig_frame: Original camera image (i.e. frame)
        """
        self.orig_frame = orig_frame

        self.lane_line_markings = None

        self.warped_frame = None
        self.transformation_matrix = None
        self.inv_transformation_matrix = None

        self.orig_image_size = self.orig_frame.shape[::-1][1:]

        width = self.orig_image_size[0]
        height = self.orig_image_size[1]
        self.width = width
        self.height = height

        self.roi_points = np.float32([
            (width / 8 * 3, height / 8 * 5),  # Top-left corner
            (width / 8 * 2, height),  # Bottom-left corner
            (width / 8 * 7, height),  # Bottom-right corner
            (width / 8 * 5, height / 8 * 5)  # Top-right corner
        ])

        self.padding = int(0.25 * width)
        self.desired_roi_points = np.float32([
            [self.padding, 0],
            [self.padding, self.orig_image_size[1]],
            [self.orig_image_size[
                 0] - self.padding, self.orig_image_size[1]],
            [self.orig_image_size[0] - self.padding, 0]
        ])

        self.histogram = None

        self.no_of_windows = 10
        self.margin = int((1 / 12) * width)
        self.minpix = int((1 / 24) * width)

        self.left_fit = None
        self.right_fit = None
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.ploty = None
        self.left_fitx = None
        self.right_fitx = None
        self.leftx = None
        self.rightx = None
        self.lefty = None
        self.righty = None


    def calculate_histogram(self, frame=None, plot=True):
        """
        Calculate the image histogram to find peaks in white pixel count

        :param frame: The warped image
        :param plot: Create a plot if True
        """
        if frame is None:
            frame = self.warped_frame

        self.histogram = np.sum(frame[int(
            frame.shape[0] / 2):, :], axis=0)

        if plot == True:
            figure, (ax1, ax2) = plt.subplots(2, 1)  # 2 row, 1 columns
            figure.set_size_inches(10, 5)
            ax1.imshow(frame, cmap='gray')
            ax1.set_title("Warped Binary Frame")
            ax2.plot(self.histogram)
            ax2.set_title("Histogram Peaks")
            plt.show()

        return self.histogram

    def get_lane_line_previous_window(self, left_fit, right_fit, plot=False):
        """
        Use the lane line from the previous sliding window to get the parameters
        for the polynomial line for filling in the lane line
        :param: left_fit Polynomial function of the left lane line
        :param: right_fit Polynomial function of the right lane line
        :param: plot To display an image or not
        """
        margin = self.margin

        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (left_fit[0] * (
                nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
                                  nonzerox < (left_fit[0] * (
                                  nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (
                nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                                   nonzerox < (right_fit[0] * (
                                   nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))
        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        self.leftx = leftx
        self.rightx = rightx
        self.lefty = lefty
        self.righty = righty

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        self.left_fit = left_fit
        self.right_fit = right_fit

        ploty = np.linspace(
            0, self.warped_frame.shape[0] - 1, self.warped_frame.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        self.ploty = ploty
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx

        if plot == True:
            out_img = np.dstack((self.warped_frame, self.warped_frame, (
                self.warped_frame))) * 255
            window_img = np.zeros_like(out_img)

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
                0, 0, 255]
            margin = self.margin
            left_line_window1 = np.array([np.transpose(np.vstack([
                left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                left_fitx + margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([
                right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                right_fitx + margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            figure, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(self.warped_frame, cmap='gray')
            ax3.imshow(result)
            ax3.plot(left_fitx, ploty, color='yellow')
            ax3.plot(right_fitx, ploty, color='yellow')
            ax1.set_title("Original Frame")
            ax2.set_title("Warped Frame")
            ax3.set_title("Warped Frame With Search Window")
            plt.show()

    def get_lane_line_indices_sliding_windows(self, plot=False):
        """
        Get the indices of the lane line pixels using the
        sliding windows technique.

        :param: plot Show plot or not
        :return: Best fit lines for the left and right lines of the current lane
        """
        margin = self.margin

        frame_sliding_window = self.warped_frame.copy()

        window_height = np.int(self.warped_frame.shape[0] / self.no_of_windows)

        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = []
        right_lane_inds = []

        leftx_base, rightx_base = self.histogram_peak()
        leftx_current = leftx_base
        rightx_current = rightx_base

        no_of_windows = self.no_of_windows

        for window in range(no_of_windows):

            win_y_low = self.warped_frame.shape[0] - (window + 1) * window_height
            win_y_high = self.warped_frame.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(frame_sliding_window, (win_xleft_low, win_y_low), (
                win_xleft_high, win_y_high), (255, 255, 255), 2)
            cv2.rectangle(frame_sliding_window, (win_xright_low, win_y_low), (
                win_xright_high, win_y_high), (255, 255, 255), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (
                                      nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (
                                       nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            minpix = self.minpix
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        self.left_fit = left_fit
        self.right_fit = right_fit

        if plot == True:
            ploty = np.linspace(
                0, frame_sliding_window.shape[0] - 1, frame_sliding_window.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            out_img = np.dstack((
                frame_sliding_window, frame_sliding_window, (
                    frame_sliding_window))) * 255

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
                0, 0, 255]

            figure, (ax1, ax2, ax3) = plt.subplots(3, 1)
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(frame_sliding_window, cmap='gray')
            ax3.imshow(out_img)
            ax3.plot(left_fitx, ploty, color='yellow')
            ax3.plot(right_fitx, ploty, color='yellow')
            ax1.set_title("Original Frame")
            ax2.set_title("Warped Frame with Sliding Windows")
            ax3.set_title("Detected Lane Lines with Sliding Windows")
            plt.show()

        return self.left_fit, self.right_fit

    def get_line_markings(self, frame=None):
        """
        Isolates lane lines.

        :param frame: The camera frame that contains the lanes we want to detect
        :return: Binary (i.e. black and white) image containing the lane lines.
        """
        if frame is None:
            frame = self.orig_frame

        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        _, sxbinary = edge.threshold(hls[:, :, 1], thresh=(120, 255))
        sxbinary = edge.blur_gaussian(sxbinary, ksize=3)

        sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))

      
        s_channel = hls[:, :, 2]
        _, s_binary = edge.threshold(s_channel, (80, 255))

        _, r_thresh = edge.threshold(frame[:, :, 2], thresh=(120, 255))

        rs_binary = cv2.bitwise_and(s_binary, r_thresh)

        self.lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary.astype(
            np.uint8))
        return self.lane_line_markings

    def histogram_peak(self):
        """
        Get the left and right peak of the histogram

        Return the x coordinate of the left histogram peak and the right histogram
        peak.
        """
        midpoint = np.int(self.histogram.shape[0] / 2)
        leftx_base = np.argmax(self.histogram[:midpoint])
        rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint

        return leftx_base, rightx_base

    def overlay_lane_lines(self, plot=False):
        """
        Overlay lane lines on the original frame
        :param: Plot the lane lines if True
        :return: Lane with overlay
        """
        warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([
            self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([
            self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        newwarp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (
            self.orig_frame.shape[
                1], self.orig_frame.shape[0]))

        result = cv2.addWeighted(self.orig_frame, 1, newwarp, 0.3, 0)

        if plot == True:
            figure, (ax1, ax2) = plt.subplots(2, 1)
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            ax1.set_title("Original Frame")
            ax2.set_title("Original Frame With Lane Overlay")
            plt.show()

        return result

    def perspective_transform(self, frame=None, plot=False):
        """
        Perform the perspective transform.
        :param: frame Current frame
        :param: plot Plot the warped image if True
        :return: Bird's eye view of the current lane
        """
        if frame is None:
            frame = self.lane_line_markings

        self.transformation_matrix = cv2.getPerspectiveTransform(
            self.roi_points, self.desired_roi_points)

        self.inv_transformation_matrix = cv2.getPerspectiveTransform(
            self.desired_roi_points, self.roi_points)

        self.warped_frame = cv2.warpPerspective(
            frame, self.transformation_matrix, self.orig_image_size, flags=(
                cv2.INTER_LINEAR))

        (thresh, binary_warped) = cv2.threshold(
            self.warped_frame, 127, 255, cv2.THRESH_BINARY)
        self.warped_frame = binary_warped

        if plot == True:
            warped_copy = self.warped_frame.copy()
            warped_plot = cv2.polylines(warped_copy, np.int32([
                self.desired_roi_points]), True, (147, 20, 255), 3)

            while (1):
                cv2.imshow(warped_plot)

                if cv2.waitKey(0):
                    break

            cv2.destroyAllWindows()

        return self.warped_frame

    def plot_roi(self, frame=None, plot=False):
        """
        Plot the region of interest on an image.
        :param: frame The current image frame
        :param: plot Plot the roi image if True
        """
        if plot == False:
            return

        if frame is None:
            frame = self.orig_frame.copy()

        this_image = cv2.polylines(frame, np.int32([
            self.roi_points]), True, (147, 20, 255), 3)

        while (1):
            cv2.imshow(this_image)

            if cv2.waitKey(0):
                break

        cv2.destroyAllWindows()


def main(original_frame):
     # Create a Lane object
    lane_obj = Lane(orig_frame=original_frame)

    # Perform thresholding to isolate lane lines
    lane_line_markings = lane_obj.get_line_markings()

    # Plot the region of interest on the image
    lane_obj.plot_roi(plot=False)

    # Perform the perspective transform to generate a bird's eye view
    # If Plot == True, show image with new region of interest
    warped_frame = lane_obj.perspective_transform(plot=False)

    # Generate the image histogram to serve as a starting point
    # for finding lane line pixels
    histogram = lane_obj.calculate_histogram(plot=False)

    # Find lane line pixels using the sliding window method
    left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(
        plot=False)

    # Fill in the lane line
    lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)

    # Overlay lines on the original frame
    frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=False)

    return frame_with_lane_lines

#change the filename with the desired picture
filename = 'solidYellowLeft.jpg'
image = cv2.imread(filename)
cv2.imshow(main(image))
cv2.waitKey(0)
cv2.destroyAllWindows()
