(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Doccer : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val docformat : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
`docformat` is deprecated!
scipy.misc.docformat is deprecated in Scipy 1.3.0
*)

val extend_notes_in_docstring : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
`extend_notes_in_docstring` is deprecated!
scipy.misc.extend_notes_in_docstring is deprecated in Scipy 1.3.0
*)

val filldoc : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
`filldoc` is deprecated!
scipy.misc.filldoc is deprecated in Scipy 1.3.0
*)

val indentcount_lines : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
`indentcount_lines` is deprecated!
scipy.misc.indentcount_lines is deprecated in Scipy 1.3.0
*)

val inherit_docstring_from : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
`inherit_docstring_from` is deprecated!
scipy.misc.inherit_docstring_from is deprecated in Scipy 1.3.0
*)

val replace_notes_in_docstring : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
`replace_notes_in_docstring` is deprecated!
scipy.misc.replace_notes_in_docstring is deprecated in Scipy 1.3.0
*)

val unindent_dict : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
`unindent_dict` is deprecated!
scipy.misc.unindent_dict is deprecated in Scipy 1.3.0
*)

val unindent_string : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
`unindent_string` is deprecated!
scipy.misc.unindent_string is deprecated in Scipy 1.3.0
*)


end

val ascent : unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Get an 8-bit grayscale bit-depth, 512 x 512 derived image for easy use in demos

The image is derived from accent-to-the-top.jpg at
http://www.public-domain-image.com/people-public-domain-images-pictures/

Parameters
----------
None

Returns
-------
ascent : ndarray
   convenient image to use for testing and demonstration

Examples
--------
>>> import scipy.misc
>>> ascent = scipy.misc.ascent()
>>> ascent.shape
(512, 512)
>>> ascent.max()
255

>>> import matplotlib.pyplot as plt
>>> plt.gray()
>>> plt.imshow(ascent)
>>> plt.show()
*)

val central_diff_weights : ?ndiv:int -> np:int -> unit -> Py.Object.t
(**
Return weights for an Np-point central derivative.

Assumes equally-spaced function points.

If weights are in the vector w, then
derivative is w[0] * f(x-ho*dx) + ... + w[-1] * f(x+h0*dx)

Parameters
----------
Np : int
    Number of points for the central derivative.
ndiv : int, optional
    Number of divisions.  Default is 1.

Notes
-----
Can be inaccurate for large number of points.
*)

val derivative : ?dx:float -> ?n:int -> ?args:Py.Object.t -> ?order:int -> func:Py.Object.t -> x0:float -> unit -> Py.Object.t
(**
Find the n-th derivative of a function at a point.

Given a function, use a central difference formula with spacing `dx` to
compute the `n`-th derivative at `x0`.

Parameters
----------
func : function
    Input function.
x0 : float
    The point at which `n`-th derivative is found.
dx : float, optional
    Spacing.
n : int, optional
    Order of the derivative. Default is 1.
args : tuple, optional
    Arguments
order : int, optional
    Number of points to use, must be odd.

Notes
-----
Decreasing the step size too small can result in round-off error.

Examples
--------
>>> from scipy.misc import derivative
>>> def f(x):
...     return x**3 + x**2
>>> derivative(f, 1.0, dx=1e-6)
4.9999999999217337
*)

val electrocardiogram : unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Load an electrocardiogram as an example for a one-dimensional signal.

The returned signal is a 5 minute long electrocardiogram (ECG), a medical
recording of the heart's electrical activity, sampled at 360 Hz.

Returns
-------
ecg : ndarray
    The electrocardiogram in millivolt (mV) sampled at 360 Hz.

Notes
-----
The provided signal is an excerpt (19:35 to 24:35) from the `record 208`_
(lead MLII) provided by the MIT-BIH Arrhythmia Database [1]_ on
PhysioNet [2]_. The excerpt includes noise induced artifacts, typical
heartbeats as well as pathological changes.

.. _record 208: https://physionet.org/physiobank/database/html/mitdbdir/records.htm#208

.. versionadded:: 1.1.0

References
----------
.. [1] Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
       IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
       (PMID: 11446209); :doi:`10.13026/C2F305`
.. [2] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh,
       Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank,
       PhysioToolkit, and PhysioNet: Components of a New Research Resource
       for Complex Physiologic Signals. Circulation 101(23):e215-e220;
       :doi:`10.1161/01.CIR.101.23.e215`

Examples
--------
>>> from scipy.misc import electrocardiogram
>>> ecg = electrocardiogram()
>>> ecg
array([-0.245, -0.215, -0.185, ..., -0.405, -0.395, -0.385])
>>> ecg.shape, ecg.mean(), ecg.std()
((108000,), -0.16510875, 0.5992473991177294)

As stated the signal features several areas with a different morphology.
E.g. the first few seconds show the electrical activity of a heart in
normal sinus rhythm as seen below.

>>> import matplotlib.pyplot as plt
>>> fs = 360
>>> time = np.arange(ecg.size) / fs
>>> plt.plot(time, ecg)
>>> plt.xlabel('time in s')
>>> plt.ylabel('ECG in mV')
>>> plt.xlim(9, 10.2)
>>> plt.ylim(-1, 1.5)
>>> plt.show()

After second 16 however, the first premature ventricular contractions, also
called extrasystoles, appear. These have a different morphology compared to
typical heartbeats. The difference can easily be observed in the following
plot.

>>> plt.plot(time, ecg)
>>> plt.xlabel('time in s')
>>> plt.ylabel('ECG in mV')
>>> plt.xlim(46.5, 50)
>>> plt.ylim(-2, 1.5)
>>> plt.show()

At several points large artifacts disturb the recording, e.g.:

>>> plt.plot(time, ecg)
>>> plt.xlabel('time in s')
>>> plt.ylabel('ECG in mV')
>>> plt.xlim(207, 215)
>>> plt.ylim(-2, 3.5)
>>> plt.show()

Finally, examining the power spectrum reveals that most of the biosignal is
made up of lower frequencies. At 60 Hz the noise induced by the mains
electricity can be clearly observed.

>>> from scipy.signal import welch
>>> f, Pxx = welch(ecg, fs=fs, nperseg=2048, scaling='spectrum')
>>> plt.semilogy(f, Pxx)
>>> plt.xlabel('Frequency in Hz')
>>> plt.ylabel('Power spectrum of the ECG in mV**2')
>>> plt.xlim(f[[0, -1]])
>>> plt.show()
*)

val face : ?gray:bool -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Get a 1024 x 768, color image of a raccoon face.

raccoon-procyon-lotor.jpg at http://www.public-domain-image.com

Parameters
----------
gray : bool, optional
    If True return 8-bit grey-scale image, otherwise return a color image

Returns
-------
face : ndarray
    image of a racoon face

Examples
--------
>>> import scipy.misc
>>> face = scipy.misc.face()
>>> face.shape
(768, 1024, 3)
>>> face.max()
255
>>> face.dtype
dtype('uint8')

>>> import matplotlib.pyplot as plt
>>> plt.gray()
>>> plt.imshow(face)
>>> plt.show()
*)

