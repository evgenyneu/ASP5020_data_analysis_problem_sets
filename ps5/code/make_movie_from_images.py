"""Create mp4 movie from images. Requires ffmpeg program to be installed."""

from shutil import move
import os
import subprocess


def create_dir(dir):
    """Creates a directory if it does not exist.

    Parameters
    ----------
    dir : str
        Directory path, can be nested directories.

    """

    if not os.path.exists(dir):
        os.makedirs(dir)


def make_movie_from_images(plot_dir, movie_dir, movie_name, frame_rate):
    """
    Creates a movie .mp4 file from individual images.

    Parameters
    ----------
    plot_dir : str
        Path to directory containing individual images for the movie.
    movie_dir : str
        The output directory of the movie
    frame_rate : int
        The frame rate of the movie, in frames per second.

    """

    src_movie = os.path.join(plot_dir, movie_name)

    if os.path.exists(src_movie):
        os.remove(src_movie)

    command = (f"ffmpeg -framerate {frame_rate} -pattern_type glob -i '*.png' "
               f"-c:v libx264 -pix_fmt yuv420p {movie_name}")

    subprocess.call(command, cwd=plot_dir, shell=True)

    # Copy movie to output directory
    # ----------

    dest_movie = os.path.join(movie_dir, movie_name)
    create_dir(movie_dir)

    if os.path.exists(dest_movie):
        os.remove(dest_movie)

    move(src_movie, dest_movie)

    print(f"Movie saved to: {dest_movie}")
