(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module FortranEOFError : sig
type tag = [`FortranEOFError]
type t = [`BaseException | `FortranEOFError | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_exception : t -> [`BaseException] Obj.t
val with_traceback : tb:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module FortranFile : sig
type tag = [`FortranFile]
type t = [`FortranFile | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?mode:[`R | `W] -> ?header_dtype:Np.Dtype.t -> filename:[`File of Py.Object.t | `S of string] -> unit -> t
(**
A file object for unformatted sequential files from Fortran code.

Parameters
----------
filename : file or str
    Open file object or filename.
mode : {'r', 'w'}, optional
    Read-write mode, default is 'r'.
header_dtype : dtype, optional
    Data type of the header. Size and endiness must match the input/output file.

Notes
-----
These files are broken up into records of unspecified types. The size of
each record is given at the start (although the size of this header is not
standard) and the data is written onto disk without any formatting. Fortran
compilers supporting the BACKSPACE statement will write a second copy of
the size to facilitate backwards seeking.

This class only supports files written with both sizes for the record.
It also does not support the subrecords used in Intel and gfortran compilers
for records which are greater than 2GB with a 4-byte header.

An example of an unformatted sequential file in Fortran would be written as::

    OPEN(1, FILE=myfilename, FORM='unformatted')

    WRITE(1) myvariable

Since this is a non-standard file format, whose contents depend on the
compiler and the endianness of the machine, caution is advised. Files from
gfortran 4.8.0 and gfortran 4.1.2 on x86_64 are known to work.

Consider using Fortran direct-access files or files from the newer Stream
I/O, which can be easily read by `numpy.fromfile`.

Examples
--------
To create an unformatted sequential Fortran file:

>>> from scipy.io import FortranFile
>>> f = FortranFile('test.unf', 'w')
>>> f.write_record(np.array([1,2,3,4,5], dtype=np.int32))
>>> f.write_record(np.linspace(0,1,20).reshape((5,4)).T)
>>> f.close()

To read this file:

>>> f = FortranFile('test.unf', 'r')
>>> print(f.read_ints(np.int32))
[1 2 3 4 5]
>>> print(f.read_reals(float).reshape((5,4), order='F'))
[[0.         0.05263158 0.10526316 0.15789474]
 [0.21052632 0.26315789 0.31578947 0.36842105]
 [0.42105263 0.47368421 0.52631579 0.57894737]
 [0.63157895 0.68421053 0.73684211 0.78947368]
 [0.84210526 0.89473684 0.94736842 1.        ]]
>>> f.close()

Or, in Fortran::

    integer :: a(5), i
    double precision :: b(5,4)
    open(1, file='test.unf', form='unformatted')
    read(1) a
    read(1) b
    close(1)
    write( *,* ) a
    do i = 1, 5
        write( *,* ) b(i,:)
    end do
*)

val close : [> tag] Obj.t -> Py.Object.t
(**
Closes the file. It is unsupported to call any other methods off this
object after closing it. Note that this class supports the 'with'
statement in modern versions of Python, to call this automatically
*)

val read_ints : ?dtype:Np.Dtype.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Reads a record of a given type from the file, defaulting to an integer
type (``INTEGER*4`` in Fortran).

Parameters
----------
dtype : dtype, optional
    Data type specifying the size and endiness of the data.

Returns
-------
data : ndarray
    A 1-D array object.

See Also
--------
read_reals
read_record
*)

val read_reals : ?dtype:Np.Dtype.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Reads a record of a given type from the file, defaulting to a floating
point number (``real*8`` in Fortran).

Parameters
----------
dtype : dtype, optional
    Data type specifying the size and endiness of the data.

Returns
-------
data : ndarray
    A 1-D array object.

See Also
--------
read_ints
read_record
*)

val read_record : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Reads a record of a given type from the file.

Parameters
----------
*dtypes : dtypes, optional
    Data type(s) specifying the size and endiness of the data.

Returns
-------
data : ndarray
    A 1-D array object.

Raises
------
FortranEOFError
    To signal that no further records are available
FortranFormattingError
    To signal that the end of the file was encountered
    part-way through a record

Notes
-----
If the record contains a multidimensional array, you can specify
the size in the dtype. For example::

    INTEGER var(5,4)

can be read with::

    read_record('(4,5)i4').T

Note that this function does **not** assume the file data is in Fortran
column major order, so you need to (i) swap the order of dimensions
when reading and (ii) transpose the resulting array.

Alternatively, you can read the data as a 1-D array and handle the
ordering yourself. For example::

    read_record('i4').reshape(5, 4, order='F')

For records that contain several variables or mixed types (as opposed
to single scalar or array types), give them as separate arguments::

    double precision :: a
    integer :: b
    write(1) a, b

    record = f.read_record('<f4', '<i4')
    a = record[0]  # first number
    b = record[1]  # second number

and if any of the variables are arrays, the shape can be specified as
the third item in the relevant dtype::

    double precision :: a
    integer :: b(3,4)
    write(1) a, b

    record = f.read_record('<f4', np.dtype(('<i4', (4, 3))))
    a = record[0]
    b = record[1].T

NumPy also supports a short syntax for this kind of type::

    record = f.read_record('<f4', '(3,3)<i4')

See Also
--------
read_reals
read_ints
*)

val write_record : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
Write a record (including sizes) to the file.

Parameters
----------
*items : array_like
    The data arrays to write.

Notes
-----
Writes data items to a file::

    write_record(a.T, b.T, c.T, ...)

    write(1) a, b, c, ...

Note that data in multidimensional arrays is written in
row-major order --- to make them read correctly by Fortran
programs, you need to transpose the arrays yourself when
writing them.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module FortranFormattingError : sig
type tag = [`FortranFormattingError]
type t = [`BaseException | `FortranFormattingError | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_exception : t -> [`BaseException] Obj.t
val with_traceback : tb:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Netcdf_file : sig
type tag = [`Netcdf_file]
type t = [`Netcdf_file | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?mode:[`R | `W | `A] -> ?mmap:bool -> ?version:[`Two | `One] -> ?maskandscale:bool -> filename:[`File_like of Py.Object.t | `S of string] -> unit -> t
(**
A file object for NetCDF data.

A `netcdf_file` object has two standard attributes: `dimensions` and
`variables`. The values of both are dictionaries, mapping dimension
names to their associated lengths and variable names to variables,
respectively. Application programs should never modify these
dictionaries.

All other attributes correspond to global attributes defined in the
NetCDF file. Global file attributes are created by assigning to an
attribute of the `netcdf_file` object.

Parameters
----------
filename : string or file-like
    string -> filename
mode : {'r', 'w', 'a'}, optional
    read-write-append mode, default is 'r'
mmap : None or bool, optional
    Whether to mmap `filename` when reading.  Default is True
    when `filename` is a file name, False when `filename` is a
    file-like object. Note that when mmap is in use, data arrays
    returned refer directly to the mmapped data on disk, and the
    file cannot be closed as long as references to it exist.
version : {1, 2}, optional
    version of netcdf to read / write, where 1 means *Classic
    format* and 2 means *64-bit offset format*.  Default is 1.  See
    `here <https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_introduction.html#select_format>`__
    for more info.
maskandscale : bool, optional
    Whether to automatically scale and/or mask data based on attributes.
    Default is False.

Notes
-----
The major advantage of this module over other modules is that it doesn't
require the code to be linked to the NetCDF libraries. This module is
derived from `pupynere <https://bitbucket.org/robertodealmeida/pupynere/>`_.

NetCDF files are a self-describing binary data format. The file contains
metadata that describes the dimensions and variables in the file. More
details about NetCDF files can be found `here
<https://www.unidata.ucar.edu/software/netcdf/guide_toc.html>`__. There
are three main sections to a NetCDF data structure:

1. Dimensions
2. Variables
3. Attributes

The dimensions section records the name and length of each dimension used
by the variables. The variables would then indicate which dimensions it
uses and any attributes such as data units, along with containing the data
values for the variable. It is good practice to include a
variable that is the same name as a dimension to provide the values for
that axes. Lastly, the attributes section would contain additional
information such as the name of the file creator or the instrument used to
collect the data.

When writing data to a NetCDF file, there is often the need to indicate the
'record dimension'. A record dimension is the unbounded dimension for a
variable. For example, a temperature variable may have dimensions of
latitude, longitude and time. If one wants to add more temperature data to
the NetCDF file as time progresses, then the temperature variable should
have the time dimension flagged as the record dimension.

In addition, the NetCDF file header contains the position of the data in
the file, so access can be done in an efficient manner without loading
unnecessary data into memory. It uses the ``mmap`` module to create
Numpy arrays mapped to the data on disk, for the same purpose.

Note that when `netcdf_file` is used to open a file with mmap=True
(default for read-only), arrays returned by it refer to data
directly on the disk. The file should not be closed, and cannot be cleanly
closed when asked, if such arrays are alive. You may want to copy data arrays
obtained from mmapped Netcdf file if they are to be processed after the file
is closed, see the example below.

Examples
--------
To create a NetCDF file:

>>> from scipy.io import netcdf
>>> f = netcdf.netcdf_file('simple.nc', 'w')
>>> f.history = 'Created for a test'
>>> f.createDimension('time', 10)
>>> time = f.createVariable('time', 'i', ('time',))
>>> time[:] = np.arange(10)
>>> time.units = 'days since 2008-01-01'
>>> f.close()

Note the assignment of ``arange(10)`` to ``time[:]``.  Exposing the slice
of the time variable allows for the data to be set in the object, rather
than letting ``arange(10)`` overwrite the ``time`` variable.

To read the NetCDF file we just created:

>>> from scipy.io import netcdf
>>> f = netcdf.netcdf_file('simple.nc', 'r')
>>> print(f.history)
b'Created for a test'
>>> time = f.variables['time']
>>> print(time.units)
b'days since 2008-01-01'
>>> print(time.shape)
(10,)
>>> print(time[-1])
9

NetCDF files, when opened read-only, return arrays that refer
directly to memory-mapped data on disk:

>>> data = time[:]
>>> data.base.base
<mmap.mmap object at 0x7fe753763180>

If the data is to be processed after the file is closed, it needs
to be copied to main memory:

>>> data = time[:].copy()
>>> f.close()
>>> data.mean()
4.5

A NetCDF file can also be used as context manager:

>>> from scipy.io import netcdf
>>> with netcdf.netcdf_file('simple.nc', 'r') as f:
...     print(f.history)
b'Created for a test'
*)

val close : [> tag] Obj.t -> Py.Object.t
(**
Closes the NetCDF file.
*)

val createDimension : name:string -> length:int -> [> tag] Obj.t -> Py.Object.t
(**
Adds a dimension to the Dimension section of the NetCDF data structure.

Note that this function merely adds a new dimension that the variables can
reference. The values for the dimension, if desired, should be added as
a variable using `createVariable`, referring to this dimension.

Parameters
----------
name : str
    Name of the dimension (Eg, 'lat' or 'time').
length : int
    Length of the dimension.

See Also
--------
createVariable
*)

val createVariable : name:string -> type_:[`S of string | `Dtype of Np.Dtype.t] -> dimensions:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Create an empty variable for the `netcdf_file` object, specifying its data
type and the dimensions it uses.

Parameters
----------
name : str
    Name of the new variable.
type : dtype or str
    Data type of the variable.
dimensions : sequence of str
    List of the dimension names used by the variable, in the desired order.

Returns
-------
variable : netcdf_variable
    The newly created ``netcdf_variable`` object.
    This object has also been added to the `netcdf_file` object as well.

See Also
--------
createDimension

Notes
-----
Any dimensions to be used by the variable should already exist in the
NetCDF data structure or should be created by `createDimension` prior to
creating the NetCDF variable.
*)

val flush : [> tag] Obj.t -> Py.Object.t
(**
Perform a sync-to-disk flush if the `netcdf_file` object is in write mode.

See Also
--------
sync : Identical function
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Netcdf_variable : sig
type tag = [`Netcdf_variable]
type t = [`Netcdf_variable | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?attributes:Py.Object.t -> ?maskandscale:bool -> data:[>`Ndarray] Np.Obj.t -> typecode:Py.Object.t -> size:int -> shape:int list -> dimensions:Py.Object.t -> unit -> t
(**
A data object for netcdf files.

`netcdf_variable` objects are constructed by calling the method
`netcdf_file.createVariable` on the `netcdf_file` object. `netcdf_variable`
objects behave much like array objects defined in numpy, except that their
data resides in a file. Data is read by indexing and written by assigning
to an indexed subset; the entire array can be accessed by the index ``[:]``
or (for scalars) by using the methods `getValue` and `assignValue`.
`netcdf_variable` objects also have attribute `shape` with the same meaning
as for arrays, but the shape cannot be modified. There is another read-only
attribute `dimensions`, whose value is the tuple of dimension names.

All other attributes correspond to variable attributes defined in
the NetCDF file. Variable attributes are created by assigning to an
attribute of the `netcdf_variable` object.

Parameters
----------
data : array_like
    The data array that holds the values for the variable.
    Typically, this is initialized as empty, but with the proper shape.
typecode : dtype character code
    Desired data-type for the data array.
size : int
    Desired element size for the data array.
shape : sequence of ints
    The shape of the array. This should match the lengths of the
    variable's dimensions.
dimensions : sequence of strings
    The names of the dimensions used by the variable. Must be in the
    same order of the dimension lengths given by `shape`.
attributes : dict, optional
    Attribute values (any type) keyed by string names. These attributes
    become attributes for the netcdf_variable object.
maskandscale : bool, optional
    Whether to automatically scale and/or mask data based on attributes.
    Default is False.


Attributes
----------
dimensions : list of str
    List of names of dimensions used by the variable object.
isrec, shape
    Properties

See also
--------
isrec, shape
*)

val __getitem__ : index:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __setitem__ : index:Py.Object.t -> data:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val assignValue : value:[`F of float | `I of int | `Bool of bool | `S of string] -> [> tag] Obj.t -> Py.Object.t
(**
Assign a scalar value to a `netcdf_variable` of length one.

Parameters
----------
value : scalar
    Scalar value (of compatible type) to assign to a length-one netcdf
    variable. This value will be written to file.

Raises
------
ValueError
    If the input is not a scalar, or if the destination is not a length-one
    netcdf variable.
*)

val getValue : [> tag] Obj.t -> Py.Object.t
(**
Retrieve a scalar value from a `netcdf_variable` of length one.

Raises
------
ValueError
    If the netcdf variable is an array of length greater than one,
    this exception will be raised.
*)

val itemsize : [> tag] Obj.t -> int
(**
Return the itemsize of the variable.

Returns
-------
itemsize : int
    The element size of the variable (e.g., 8 for float64).
*)

val typecode : [> tag] Obj.t -> Py.Object.t
(**
Return the typecode of the variable.

Returns
-------
typecode : char
    The character typecode of the variable (e.g., 'i' for int).
*)


(** Attribute dimensions: get value or raise Not_found if None.*)
val dimensions : t -> string list

(** Attribute dimensions: get value as an option. *)
val dimensions_opt : t -> (string list) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Harwell_boeing : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module HBFile : sig
type tag = [`HBFile]
type t = [`HBFile | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?hb_info:Py.Object.t -> file:Py.Object.t -> unit -> t
(**
None
*)

val read_matrix : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val write_matrix : m:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module HBInfo : sig
type tag = [`HBInfo]
type t = [`HBInfo | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?right_hand_sides_nlines:Py.Object.t -> ?nelementals:Py.Object.t -> title:Py.Object.t -> key:Py.Object.t -> total_nlines:Py.Object.t -> pointer_nlines:Py.Object.t -> indices_nlines:Py.Object.t -> values_nlines:Py.Object.t -> mxtype:Py.Object.t -> nrows:Py.Object.t -> ncols:Py.Object.t -> nnon_zeros:Py.Object.t -> pointer_format_str:Py.Object.t -> indices_format_str:Py.Object.t -> values_format_str:Py.Object.t -> unit -> t
(**
None
*)

val dump : [> tag] Obj.t -> Py.Object.t
(**
Gives the header corresponding to this instance as a string.
*)

val from_data : ?title:string -> ?key:string -> ?mxtype:Py.Object.t -> ?fmt:Py.Object.t -> m:[>`Spmatrix] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Create a HBInfo instance from an existing sparse matrix.

Parameters
----------
m : sparse matrix
    the HBInfo instance will derive its parameters from m
title : str
    Title to put in the HB header
key : str
    Key
mxtype : HBMatrixType
    type of the input matrix
fmt : dict
    not implemented

Returns
-------
hb_info : HBInfo instance
*)

val from_file : fid:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Create a HBInfo instance from a file object containing a matrix in the
HB format.

Parameters
----------
fid : file-like matrix
    File or file-like object containing a matrix in the HB format.

Returns
-------
hb_info : HBInfo instance
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module HBMatrixType : sig
type tag = [`HBMatrixType]
type t = [`HBMatrixType | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?storage:Py.Object.t -> value_type:Py.Object.t -> structure:Py.Object.t -> unit -> t
(**
Class to hold the matrix type.
*)

val from_fortran : fmt:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MalformedHeader : sig
type tag = [`MalformedHeader]
type t = [`BaseException | `MalformedHeader | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_exception : t -> [`BaseException] Obj.t
val with_traceback : tb:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Hb : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module ExpFormat : sig
type tag = [`ExpFormat]
type t = [`ExpFormat | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?min:Py.Object.t -> ?repeat:Py.Object.t -> width:Py.Object.t -> significand:Py.Object.t -> unit -> t
(**
None
*)

val from_number : ?min:int -> n:float -> [> tag] Obj.t -> Py.Object.t
(**
Given a float number, returns a 'reasonable' ExpFormat instance to
represent any number between -n and n.

Parameters
----------
n : float
    max number one wants to be able to represent
min : int
    minimum number of characters to use for the format

Returns
-------
res : ExpFormat
    ExpFormat instance with reasonable (see Notes) computed width

Notes
-----
Reasonable should be understood as the minimal string length necessary
to avoid losing precision.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module FortranFormatParser : sig
type tag = [`FortranFormatParser]
type t = [`FortranFormatParser | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Parser for Fortran format strings. The parse method returns a *Format
instance.

Notes
-----
Only ExpFormat (exponential format for floating values) and IntFormat
(integer format) for now.
*)

val parse : s:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module IntFormat : sig
type tag = [`IntFormat]
type t = [`IntFormat | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?min:Py.Object.t -> ?repeat:Py.Object.t -> width:Py.Object.t -> unit -> t
(**
None
*)

val from_number : ?min:int -> n:int -> [> tag] Obj.t -> Py.Object.t
(**
Given an integer, returns a 'reasonable' IntFormat instance to represent
any number between 0 and n if n > 0, -n and n if n < 0

Parameters
----------
n : int
    max number one wants to be able to represent
min : int
    minimum number of characters to use for the format

Returns
-------
res : IntFormat
    IntFormat instance with reasonable (see Notes) computed width

Notes
-----
Reasonable should be understood as the minimal string length necessary
without losing precision. For example, IntFormat.from_number(1) will
return an IntFormat instance of width 2, so that any 0 and 1 may be
represented as 1-character strings without loss of information.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module LineOverflow : sig
type tag = [`LineOverflow]
type t = [`BaseException | `LineOverflow | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_exception : t -> [`BaseException] Obj.t
val with_traceback : tb:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Csc_matrix : sig
type tag = [`Csc_matrix]
type t = [`ArrayLike | `Csc_matrix | `IndexMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_index : t -> [`IndexMixin] Obj.t
val create : ?shape:Py.Object.t -> ?dtype:Py.Object.t -> ?copy:Py.Object.t -> arg1:Py.Object.t -> unit -> t
(**
Compressed Sparse Column matrix

This can be instantiated in several ways:

    csc_matrix(D)
        with a dense matrix or rank-2 ndarray D

    csc_matrix(S)
        with another sparse matrix S (equivalent to S.tocsc())

    csc_matrix((M, N), [dtype])
        to construct an empty matrix with shape (M, N)
        dtype is optional, defaulting to dtype='d'.

    csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
        where ``data``, ``row_ind`` and ``col_ind`` satisfy the
        relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

    csc_matrix((data, indices, indptr), [shape=(M, N)])
        is the standard CSC representation where the row indices for
        column i are stored in ``indices[indptr[i]:indptr[i+1]]``
        and their corresponding values are stored in
        ``data[indptr[i]:indptr[i+1]]``.  If the shape parameter is
        not supplied, the matrix dimensions are inferred from
        the index arrays.

Attributes
----------
dtype : dtype
    Data type of the matrix
shape : 2-tuple
    Shape of the matrix
ndim : int
    Number of dimensions (this is always 2)
nnz
    Number of stored values, including explicit zeros
data
    Data array of the matrix
indices
    CSC format index array
indptr
    CSC format index pointer array
has_sorted_indices
    Whether indices are sorted

Notes
-----

Sparse matrices can be used in arithmetic operations: they support
addition, subtraction, multiplication, division, and matrix power.

Advantages of the CSC format
    - efficient arithmetic operations CSC + CSC, CSC * CSC, etc.
    - efficient column slicing
    - fast matrix vector products (CSR, BSR may be faster)

Disadvantages of the CSC format
  - slow row slicing operations (consider CSR)
  - changes to the sparsity structure are expensive (consider LIL or DOK)


Examples
--------

>>> import numpy as np
>>> from scipy.sparse import csc_matrix
>>> csc_matrix((3, 4), dtype=np.int8).toarray()
array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]], dtype=int8)

>>> row = np.array([0, 2, 2, 0, 1, 2])
>>> col = np.array([0, 0, 1, 2, 2, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> csc_matrix((data, (row, col)), shape=(3, 3)).toarray()
array([[1, 0, 4],
       [0, 0, 5],
       [2, 3, 6]])

>>> indptr = np.array([0, 2, 3, 6])
>>> indices = np.array([0, 2, 2, 0, 1, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> csc_matrix((data, indices, indptr), shape=(3, 3)).toarray()
array([[1, 0, 4],
       [0, 0, 5],
       [2, 3, 6]])
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __setitem__ : key:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val arcsin : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arcsin.

See `numpy.arcsin` for more information.
*)

val arcsinh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arcsinh.

See `numpy.arcsinh` for more information.
*)

val arctan : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arctan.

See `numpy.arctan` for more information.
*)

val arctanh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arctanh.

See `numpy.arctanh` for more information.
*)

val argmax : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return indices of maximum elements along an axis.

Implicit zero elements are also taken into account. If there are
several maximum values, the index of the first occurrence is returned.

Parameters
----------
axis : {-2, -1, 0, 1, None}, optional
    Axis along which the argmax is computed. If None (default), index
    of the maximum element in the flatten data is returned.
out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
ind : numpy.matrix or int
    Indices of maximum elements. If matrix, its size along `axis` is 1.
*)

val argmin : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return indices of minimum elements along an axis.

Implicit zero elements are also taken into account. If there are
several minimum values, the index of the first occurrence is returned.

Parameters
----------
axis : {-2, -1, 0, 1, None}, optional
    Axis along which the argmin is computed. If None (default), index
    of the minimum element in the flatten data is returned.
out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
 ind : numpy.matrix or int
    Indices of minimum elements. If matrix, its size along `axis` is 1.
*)

val asformat : ?copy:bool -> format:[`S of string | `None] -> [> tag] Obj.t -> Py.Object.t
(**
Return this matrix in the passed format.

Parameters
----------
format : {str, None}
    The desired matrix format ('csr', 'csc', 'lil', 'dok', 'array', ...)
    or None for no conversion.
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : This matrix in the passed format.
*)

val asfptype : [> tag] Obj.t -> Py.Object.t
(**
Upcast matrix to a floating point format (if necessary)
*)

val astype : ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
(**
Cast the matrix elements to a specified type.

Parameters
----------
dtype : string or numpy dtype
    Typecode or data-type to which to cast the data.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur.
    Defaults to 'unsafe' for backwards compatibility.
    'no' means the data types should not be cast at all.
    'equiv' means only byte-order changes are allowed.
    'safe' means only casts which can preserve values are allowed.
    'same_kind' means only safe casts or casts within a kind,
    like float64 to float32, are allowed.
    'unsafe' means any data conversions may be done.
copy : bool, optional
    If `copy` is `False`, the result might share some memory with this
    matrix. If `copy` is `True`, it is guaranteed that the result and
    this matrix do not share any memory.
*)

val ceil : [> tag] Obj.t -> Py.Object.t
(**
Element-wise ceil.

See `numpy.ceil` for more information.
*)

val check_format : ?full_check:bool -> [> tag] Obj.t -> Py.Object.t
(**
check whether the matrix format is valid

Parameters
----------
full_check : bool, optional
    If `True`, rigorous check, O(N) operations. Otherwise
    basic check, O(1) operations (default True).
*)

val conj : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise complex conjugation.

If the matrix is of non-complex data type and `copy` is False,
this method does nothing and the data is not copied.

Parameters
----------
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : The element-wise complex conjugate.
*)

val conjugate : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise complex conjugation.

If the matrix is of non-complex data type and `copy` is False,
this method does nothing and the data is not copied.

Parameters
----------
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : The element-wise complex conjugate.
*)

val copy : [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of this matrix.

No data/indices will be shared between the returned value and current
matrix.
*)

val count_nonzero : [> tag] Obj.t -> Py.Object.t
(**
Number of non-zero entries, equivalent to

np.count_nonzero(a.toarray())

Unlike getnnz() and the nnz property, which return the number of stored
entries (the length of the data attribute), this method counts the
actual number of non-zero entries in data.
*)

val deg2rad : [> tag] Obj.t -> Py.Object.t
(**
Element-wise deg2rad.

See `numpy.deg2rad` for more information.
*)

val diagonal : ?k:int -> [> tag] Obj.t -> Py.Object.t
(**
Returns the kth diagonal of the matrix.

Parameters
----------
k : int, optional
    Which diagonal to get, corresponding to elements a[i, i+k].
    Default: 0 (the main diagonal).

    .. versionadded:: 1.0

See also
--------
numpy.diagonal : Equivalent numpy function.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> A.diagonal()
array([1, 0, 5])
>>> A.diagonal(k=1)
array([2, 3])
*)

val dot : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Ordinary dot product

Examples
--------
>>> import numpy as np
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> v = np.array([1, 0, -1])
>>> A.dot(v)
array([ 1, -3, -1], dtype=int64)
*)

val eliminate_zeros : [> tag] Obj.t -> Py.Object.t
(**
Remove zero entries from the matrix

This is an *in place* operation
*)

val expm1 : [> tag] Obj.t -> Py.Object.t
(**
Element-wise expm1.

See `numpy.expm1` for more information.
*)

val floor : [> tag] Obj.t -> Py.Object.t
(**
Element-wise floor.

See `numpy.floor` for more information.
*)

val getH : [> tag] Obj.t -> Py.Object.t
(**
Return the Hermitian transpose of this matrix.

See Also
--------
numpy.matrix.getH : NumPy's implementation of `getH` for matrices
*)

val get_shape : [> tag] Obj.t -> Py.Object.t
(**
Get shape of a matrix.
*)

val getcol : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of column i of the matrix, as a (m x 1)
CSC matrix (column vector).
*)

val getformat : [> tag] Obj.t -> Py.Object.t
(**
Format of a matrix representation as a string.
*)

val getmaxprint : [> tag] Obj.t -> Py.Object.t
(**
Maximum number of elements to display when printed.
*)

val getnnz : ?axis:[`One | `Zero] -> [> tag] Obj.t -> Py.Object.t
(**
Number of stored values, including explicit zeros.

Parameters
----------
axis : None, 0, or 1
    Select between the number of values across the whole matrix, in
    each column, or in each row.

See also
--------
count_nonzero : Number of non-zero entries
*)

val getrow : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of row i of the matrix, as a (1 x n)
CSR matrix (row vector).
*)

val log1p : [> tag] Obj.t -> Py.Object.t
(**
Element-wise log1p.

See `numpy.log1p` for more information.
*)

val max : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the maximum of the matrix or maximum along an axis.
This takes all elements into account, not just the non-zero ones.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the maximum over all the matrix elements, returning
    a scalar (i.e., `axis` = `None`).

out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except
    for the default value, as this argument is not used.

Returns
-------
amax : coo_matrix or scalar
    Maximum of `a`. If `axis` is None, the result is a scalar value.
    If `axis` is given, the result is a sparse.coo_matrix of dimension
    ``a.ndim - 1``.

See Also
--------
min : The minimum value of a sparse matrix along a given axis.
numpy.matrix.max : NumPy's implementation of 'max' for matrices
*)

val maximum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise maximum between this and another matrix.
*)

val mean : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the arithmetic mean along the specified axis.

Returns the average of the matrix elements. The average is taken
over all elements in the matrix by default, otherwise over the
specified axis. `float64` intermediate and return values are used
for integer inputs.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the mean is computed. The default is to compute
    the mean of all elements in the matrix (i.e., `axis` = `None`).
dtype : data-type, optional
    Type to use in computing the mean. For integer inputs, the default
    is `float64`; for floating point inputs, it is the same as the
    input dtype.

    .. versionadded:: 0.18.0

out : np.matrix, optional
    Alternative output matrix in which to place the result. It must
    have the same shape as the expected output, but the type of the
    output values will be cast if necessary.

    .. versionadded:: 0.18.0

Returns
-------
m : np.matrix

See Also
--------
numpy.matrix.mean : NumPy's implementation of 'mean' for matrices
*)

val min : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the minimum of the matrix or maximum along an axis.
This takes all elements into account, not just the non-zero ones.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the minimum over all the matrix elements, returning
    a scalar (i.e., `axis` = `None`).

out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
amin : coo_matrix or scalar
    Minimum of `a`. If `axis` is None, the result is a scalar value.
    If `axis` is given, the result is a sparse.coo_matrix of dimension
    ``a.ndim - 1``.

See Also
--------
max : The maximum value of a sparse matrix along a given axis.
numpy.matrix.min : NumPy's implementation of 'min' for matrices
*)

val minimum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise minimum between this and another matrix.
*)

val multiply : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Point-wise multiplication by another matrix, vector, or
scalar.
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
(**
nonzero indices

Returns a tuple of arrays (row,col) containing the indices
of the non-zero elements of the matrix.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1,2,0],[0,0,3],[4,0,5]])
>>> A.nonzero()
(array([0, 0, 1, 2, 2]), array([0, 1, 2, 0, 2]))
*)

val power : ?dtype:Py.Object.t -> n:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
This function performs element-wise power.

Parameters
----------
n : n is a scalar

dtype : If dtype is not specified, the current dtype will be preserved.
*)

val prune : [> tag] Obj.t -> Py.Object.t
(**
Remove empty space after all non-zero elements.
        
*)

val rad2deg : [> tag] Obj.t -> Py.Object.t
(**
Element-wise rad2deg.

See `numpy.rad2deg` for more information.
*)

val reshape : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
reshape(self, shape, order='C', copy=False)

Gives a new shape to a sparse matrix without changing its data.

Parameters
----------
shape : length-2 tuple of ints
    The new shape should be compatible with the original shape.
order : {'C', 'F'}, optional
    Read the elements using this index order. 'C' means to read and
    write the elements using C-like index order; e.g., read entire first
    row, then second row, etc. 'F' means to read and write the elements
    using Fortran-like index order; e.g., read entire first column, then
    second column, etc.
copy : bool, optional
    Indicates whether or not attributes of self should be copied
    whenever possible. The degree to which attributes are copied varies
    depending on the type of sparse matrix being used.

Returns
-------
reshaped_matrix : sparse matrix
    A sparse matrix with the given `shape`, not necessarily of the same
    format as the current object.

See Also
--------
numpy.matrix.reshape : NumPy's implementation of 'reshape' for
                       matrices
*)

val resize : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
Resize the matrix in-place to dimensions given by ``shape``

Any elements that lie within the new shape will remain at the same
indices, while non-zero elements lying outside the new shape are
removed.

Parameters
----------
shape : (int, int)
    number of rows and columns in the new matrix

Notes
-----
The semantics are not identical to `numpy.ndarray.resize` or
`numpy.resize`. Here, the same data will be maintained at each index
before and after reshape, if that index is within the new bounds. In
numpy, resizing maintains contiguity of the array, moving elements
around in the logical matrix but not within a flattened representation.

We give no guarantees about whether the underlying data attributes
(arrays, etc.) will be modified in place or replaced with new objects.
*)

val rint : [> tag] Obj.t -> Py.Object.t
(**
Element-wise rint.

See `numpy.rint` for more information.
*)

val set_shape : shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
See `reshape`.
*)

val setdiag : ?k:int -> values:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Set diagonal or off-diagonal elements of the array.

Parameters
----------
values : array_like
    New values of the diagonal elements.

    Values may have any length. If the diagonal is longer than values,
    then the remaining diagonal entries will not be set. If values if
    longer than the diagonal, then the remaining values are ignored.

    If a scalar value is given, all of the diagonal is set to it.

k : int, optional
    Which off-diagonal to set, corresponding to elements a[i,i+k].
    Default: 0 (the main diagonal).
*)

val sign : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sign.

See `numpy.sign` for more information.
*)

val sin : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sin.

See `numpy.sin` for more information.
*)

val sinh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sinh.

See `numpy.sinh` for more information.
*)

val sort_indices : [> tag] Obj.t -> Py.Object.t
(**
Sort the indices of this matrix *in place*
        
*)

val sorted_indices : [> tag] Obj.t -> Py.Object.t
(**
Return a copy of this matrix with sorted indices
        
*)

val sqrt : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sqrt.

See `numpy.sqrt` for more information.
*)

val sum : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Sum the matrix elements over a given axis.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the sum of all the matrix elements, returning a scalar
    (i.e., `axis` = `None`).
dtype : dtype, optional
    The type of the returned matrix and of the accumulator in which
    the elements are summed.  The dtype of `a` is used by default
    unless `a` has an integer dtype of less precision than the default
    platform integer.  In that case, if `a` is signed then the platform
    integer is used while if `a` is unsigned then an unsigned integer
    of the same precision as the platform integer is used.

    .. versionadded:: 0.18.0

out : np.matrix, optional
    Alternative output matrix in which to place the result. It must
    have the same shape as the expected output, but the type of the
    output values will be cast if necessary.

    .. versionadded:: 0.18.0

Returns
-------
sum_along_axis : np.matrix
    A matrix with the same shape as `self`, with the specified
    axis removed.

See Also
--------
numpy.matrix.sum : NumPy's implementation of 'sum' for matrices
*)

val sum_duplicates : [> tag] Obj.t -> Py.Object.t
(**
Eliminate duplicate matrix entries by adding them together

The is an *in place* operation
*)

val tan : [> tag] Obj.t -> Py.Object.t
(**
Element-wise tan.

See `numpy.tan` for more information.
*)

val tanh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise tanh.

See `numpy.tanh` for more information.
*)

val toarray : ?order:[`C | `F] -> ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `T2_D of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
Return a dense ndarray representation of this matrix.

Parameters
----------
order : {'C', 'F'}, optional
    Whether to store multidimensional data in C (row-major)
    or Fortran (column-major) order in memory. The default
    is 'None', indicating the NumPy default of C-ordered.
    Cannot be specified in conjunction with the `out`
    argument.

out : ndarray, 2-D, optional
    If specified, uses this array as the output buffer
    instead of allocating a new array to return. The provided
    array must have the same shape and dtype as the sparse
    matrix on which you are calling the method. For most
    sparse types, `out` is required to be memory contiguous
    (either C or Fortran ordered).

Returns
-------
arr : ndarray, 2-D
    An array with the same shape and containing the same
    data represented by the sparse matrix, with the requested
    memory order. If `out` was passed, the same object is
    returned after being modified in-place to contain the
    appropriate values.
*)

val tobsr : ?blocksize:Py.Object.t -> ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Block Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant bsr_matrix.

When blocksize=(R, C) is provided, it will be used for construction of
the bsr_matrix.
*)

val tocoo : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to COOrdinate format.

With copy=False, the data/indices may be shared between this matrix and
the resultant coo_matrix.
*)

val tocsc : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Column format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csc_matrix.
*)

val tocsr : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csr_matrix.
*)

val todense : ?order:[`C | `F] -> ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `T2_D of Py.Object.t] -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return a dense matrix representation of this matrix.

Parameters
----------
order : {'C', 'F'}, optional
    Whether to store multi-dimensional data in C (row-major)
    or Fortran (column-major) order in memory. The default
    is 'None', indicating the NumPy default of C-ordered.
    Cannot be specified in conjunction with the `out`
    argument.

out : ndarray, 2-D, optional
    If specified, uses this array (or `numpy.matrix`) as the
    output buffer instead of allocating a new array to
    return. The provided array must have the same shape and
    dtype as the sparse matrix on which you are calling the
    method.

Returns
-------
arr : numpy.matrix, 2-D
    A NumPy matrix object with the same shape and containing
    the same data represented by the sparse matrix, with the
    requested memory order. If `out` was passed and was an
    array (rather than a `numpy.matrix`), it will be filled
    with the appropriate values and returned wrapped in a
    `numpy.matrix` object that shares the same memory.
*)

val todia : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to sparse DIAgonal format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dia_matrix.
*)

val todok : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Dictionary Of Keys format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dok_matrix.
*)

val tolil : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to List of Lists format.

With copy=False, the data/indices may be shared between this matrix and
the resultant lil_matrix.
*)

val transpose : ?axes:Py.Object.t -> ?copy:bool -> [> tag] Obj.t -> Py.Object.t
(**
Reverses the dimensions of the sparse matrix.

Parameters
----------
axes : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except
    for the default value.
copy : bool, optional
    Indicates whether or not attributes of `self` should be
    copied whenever possible. The degree to which attributes
    are copied varies depending on the type of sparse matrix
    being used.

Returns
-------
p : `self` with the dimensions reversed.

See Also
--------
numpy.matrix.transpose : NumPy's implementation of 'transpose'
                         for matrices
*)

val trunc : [> tag] Obj.t -> Py.Object.t
(**
Element-wise trunc.

See `numpy.trunc` for more information.
*)


(** Attribute dtype: get value or raise Not_found if None.*)
val dtype : t -> Np.Dtype.t

(** Attribute dtype: get value as an option. *)
val dtype_opt : t -> (Np.Dtype.t) option


(** Attribute shape: get value or raise Not_found if None.*)
val shape : t -> Py.Object.t

(** Attribute shape: get value as an option. *)
val shape_opt : t -> (Py.Object.t) option


(** Attribute ndim: get value or raise Not_found if None.*)
val ndim : t -> int

(** Attribute ndim: get value as an option. *)
val ndim_opt : t -> (int) option


(** Attribute nnz: get value or raise Not_found if None.*)
val nnz : t -> Py.Object.t

(** Attribute nnz: get value as an option. *)
val nnz_opt : t -> (Py.Object.t) option


(** Attribute data: get value or raise Not_found if None.*)
val data : t -> Py.Object.t

(** Attribute data: get value as an option. *)
val data_opt : t -> (Py.Object.t) option


(** Attribute indices: get value or raise Not_found if None.*)
val indices : t -> Py.Object.t

(** Attribute indices: get value as an option. *)
val indices_opt : t -> (Py.Object.t) option


(** Attribute indptr: get value or raise Not_found if None.*)
val indptr : t -> Py.Object.t

(** Attribute indptr: get value as an option. *)
val indptr_opt : t -> (Py.Object.t) option


(** Attribute has_sorted_indices: get value or raise Not_found if None.*)
val has_sorted_indices : t -> Py.Object.t

(** Attribute has_sorted_indices: get value as an option. *)
val has_sorted_indices_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val hb_read : Py.Object.t -> Py.Object.t
(**
Read HB-format file.

Parameters
----------
path_or_open_file : path-like or file-like
    If a file-like object, it is used as-is. Otherwise, it is opened
    before reading.

Returns
-------
data : scipy.sparse.csc_matrix instance
    The data read from the HB file as a sparse matrix.

Notes
-----
At the moment not the full Harwell-Boeing format is supported. Supported
features are:

    - assembled, non-symmetric, real matrices
    - integer for pointer/indices
    - exponential format for float values, and int format

Examples
--------
We can read and write a harwell-boeing format file:

>>> from scipy.io.harwell_boeing import hb_read, hb_write
>>> from scipy.sparse import csr_matrix, eye
>>> data = csr_matrix(eye(3))  # create a sparse matrix
>>> hb_write('data.hb', data)  # write a hb file
>>> print(hb_read('data.hb'))  # read a hb file
  (0, 0)    1.0
  (1, 1)    1.0
  (2, 2)    1.0
*)

val hb_write : ?hb_info:Py.Object.t -> path_or_open_file:Py.Object.t -> m:[>`Spmatrix] Np.Obj.t -> unit -> Py.Object.t
(**
Write HB-format file.

Parameters
----------
path_or_open_file : path-like or file-like
    If a file-like object, it is used as-is. Otherwise, it is opened
    before writing.
m : sparse-matrix
    the sparse matrix to write
hb_info : HBInfo
    contains the meta-data for write

Returns
-------
None

Notes
-----
At the moment not the full Harwell-Boeing format is supported. Supported
features are:

    - assembled, non-symmetric, real matrices
    - integer for pointer/indices
    - exponential format for float values, and int format

Examples
--------
We can read and write a harwell-boeing format file:

>>> from scipy.io.harwell_boeing import hb_read, hb_write
>>> from scipy.sparse import csr_matrix, eye
>>> data = csr_matrix(eye(3))  # create a sparse matrix
>>> hb_write('data.hb', data)  # write a hb file
>>> print(hb_read('data.hb'))  # read a hb file
  (0, 0)    1.0
  (1, 1)    1.0
  (2, 2)    1.0
*)


end

val hb_read : Py.Object.t -> Py.Object.t
(**
Read HB-format file.

Parameters
----------
path_or_open_file : path-like or file-like
    If a file-like object, it is used as-is. Otherwise, it is opened
    before reading.

Returns
-------
data : scipy.sparse.csc_matrix instance
    The data read from the HB file as a sparse matrix.

Notes
-----
At the moment not the full Harwell-Boeing format is supported. Supported
features are:

    - assembled, non-symmetric, real matrices
    - integer for pointer/indices
    - exponential format for float values, and int format

Examples
--------
We can read and write a harwell-boeing format file:

>>> from scipy.io.harwell_boeing import hb_read, hb_write
>>> from scipy.sparse import csr_matrix, eye
>>> data = csr_matrix(eye(3))  # create a sparse matrix
>>> hb_write('data.hb', data)  # write a hb file
>>> print(hb_read('data.hb'))  # read a hb file
  (0, 0)    1.0
  (1, 1)    1.0
  (2, 2)    1.0
*)

val hb_write : ?hb_info:Py.Object.t -> path_or_open_file:Py.Object.t -> m:[>`Spmatrix] Np.Obj.t -> unit -> Py.Object.t
(**
Write HB-format file.

Parameters
----------
path_or_open_file : path-like or file-like
    If a file-like object, it is used as-is. Otherwise, it is opened
    before writing.
m : sparse-matrix
    the sparse matrix to write
hb_info : HBInfo
    contains the meta-data for write

Returns
-------
None

Notes
-----
At the moment not the full Harwell-Boeing format is supported. Supported
features are:

    - assembled, non-symmetric, real matrices
    - integer for pointer/indices
    - exponential format for float values, and int format

Examples
--------
We can read and write a harwell-boeing format file:

>>> from scipy.io.harwell_boeing import hb_read, hb_write
>>> from scipy.sparse import csr_matrix, eye
>>> data = csr_matrix(eye(3))  # create a sparse matrix
>>> hb_write('data.hb', data)  # write a hb file
>>> print(hb_read('data.hb'))  # read a hb file
  (0, 0)    1.0
  (1, 1)    1.0
  (2, 2)    1.0
*)


end

module Idl : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module AttrDict : sig
type tag = [`AttrDict]
type t = [`AttrDict | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?init:Py.Object.t -> unit -> t
(**
A case-insensitive dictionary with access via item, attribute, and call
notations:

    >>> d = AttrDict()
    >>> d['Variable'] = 123
    >>> d['Variable']
    123
    >>> d.Variable
    123
    >>> d.variable
    123
    >>> d('VARIABLE')
    123
*)

val __getitem__ : name:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
x.__getitem__(y) <==> x[y]
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)

val __setitem__ : key:Py.Object.t -> value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Set self[key] to value.
*)

val fromkeys : ?value:Py.Object.t -> iterable:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Create a new dictionary with keys from iterable and values set to value.
*)

val get : ?default:Py.Object.t -> key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the value for key if key is in the dictionary, else default.
*)

val pop : ?d:Py.Object.t -> k:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
If key is not found, d is returned if given, otherwise KeyError is raised
*)

val popitem : [> tag] Obj.t -> Py.Object.t
(**
Remove and return a (key, value) pair as a 2-tuple.

Pairs are returned in LIFO (last-in, first-out) order.
Raises KeyError if the dict is empty.
*)

val setdefault : ?default:Py.Object.t -> key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Insert key with a value of default if key is not in the dictionary.

Return the value for key if key is in the dictionary, else default.
*)

val update : ?e:Py.Object.t -> ?f:(string * Py.Object.t) list -> [> tag] Obj.t -> Py.Object.t
(**
D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
In either case, this is followed by: for k in F:  D[k] = F[k]
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ObjectPointer : sig
type tag = [`ObjectPointer]
type t = [`Object | `ObjectPointer] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t -> t
(**
Class used to define object pointers
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Pointer : sig
type tag = [`Pointer]
type t = [`Object | `Pointer] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t -> t
(**
Class used to define pointers
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val asstr : Py.Object.t -> Py.Object.t
(**
None
*)

val readsav : ?idict:Py.Object.t -> ?python_dict:bool -> ?uncompressed_file_name:string -> ?verbose:bool -> file_name:string -> unit -> Py.Object.t
(**
Read an IDL .sav file.

Parameters
----------
file_name : str
    Name of the IDL save file.
idict : dict, optional
    Dictionary in which to insert .sav file variables.
python_dict : bool, optional
    By default, the object return is not a Python dictionary, but a
    case-insensitive dictionary with item, attribute, and call access
    to variables. To get a standard Python dictionary, set this option
    to True.
uncompressed_file_name : str, optional
    This option only has an effect for .sav files written with the
    /compress option. If a file name is specified, compressed .sav
    files are uncompressed to this file. Otherwise, readsav will use
    the `tempfile` module to determine a temporary filename
    automatically, and will remove the temporary file upon successfully
    reading it in.
verbose : bool, optional
    Whether to print out information about the save file, including
    the records read, and available variables.

Returns
-------
idl_dict : AttrDict or dict
    If `python_dict` is set to False (default), this function returns a
    case-insensitive dictionary with item, attribute, and call access
    to variables. If `python_dict` is set to True, this function
    returns a Python dictionary with all variable names in lowercase.
    If `idict` was specified, then variables are written to the
    dictionary specified, and the updated dictionary is returned.

Examples
--------
>>> from os.path import dirname, join as pjoin
>>> import scipy.io as sio
>>> from scipy.io import readsav

Get the filename for an example .sav file from the tests/data directory.

>>> data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
>>> sav_fname = pjoin(data_dir, 'array_float32_1d.sav')

Load the .sav file contents.

>>> sav_data = readsav(sav_fname)

Get keys of the .sav file contents.

>>> print(sav_data.keys())
dict_keys(['array1d'])

Access a content with a key.

>>> print(sav_data['array1d'])
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0.]
*)


end

module Matlab : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Mio : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module MatFile4Reader : sig
type tag = [`MatFile4Reader]
type t = [`MatFile4Reader | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> mat_stream:Py.Object.t -> Py.Object.t list -> t
(**
Reader for Mat4 files 
*)

val end_of_stream : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val get_variables : ?variable_names:[`Sequence_of_str of Py.Object.t | `S of string] -> [> tag] Obj.t -> Py.Object.t
(**
get variables from stream as dictionary

Parameters
----------
variable_names : None or str or sequence of str, optional
    variable name, or sequence of variable names to get from Mat file /
    file stream. If None, then get all variables in file.
*)

val guess_byte_order : [> tag] Obj.t -> Py.Object.t
(**
As we do not know what file type we have, assume native 
*)

val initialize_read : [> tag] Obj.t -> Py.Object.t
(**
Run when beginning read of variables

Sets up readers from parameters in `self`
*)

val list_variables : [> tag] Obj.t -> Py.Object.t
(**
list variables from stream 
*)

val read_var_array : ?process:bool -> header:Py.Object.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Read array, given `header`

Parameters
----------
header : header object
   object with fields defining variable header
process : {True, False}, optional
   If True, apply recursive post-processing during loading of array.

Returns
-------
arr : array
   array with post-processing applied or not according to
   `process`.
*)

val read_var_header : [> tag] Obj.t -> (Py.Object.t * int)
(**
Read and return header, next position

Parameters
----------
None

Returns
-------
header : object
   object that can be passed to self.read_var_array, and that
   has attributes ``name`` and ``is_global``
next_position : int
   position in stream of next variable
*)

val set_matlab_compatible : [> tag] Obj.t -> Py.Object.t
(**
Sets options to return arrays as MATLAB loads them 
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MatFile4Writer : sig
type tag = [`MatFile4Writer]
type t = [`MatFile4Writer | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?oned_as:Py.Object.t -> file_stream:Py.Object.t -> unit -> t
(**
Class for writing matlab 4 format files 
*)

val put_variables : ?write_header:bool -> mdict:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Write variables in `mdict` to stream

Parameters
----------
mdict : mapping
   mapping with method ``items`` return name, contents pairs
   where ``name`` which will appeak in the matlab workspace in
   file load, and ``contents`` is something writeable to a
   matlab file, such as a NumPy array.
write_header : {None, True, False}
   If True, then write the matlab file header before writing the
   variables. If None (the default) then write the file header
   if we are at position 0 in the stream. By setting False
   here, and setting the stream position to the end of the file,
   you can append variables to a matlab file
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MatFile5Reader : sig
type tag = [`MatFile5Reader]
type t = [`MatFile5Reader | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?byte_order:Py.Object.t -> ?mat_dtype:Py.Object.t -> ?squeeze_me:Py.Object.t -> ?chars_as_strings:Py.Object.t -> ?matlab_compatible:Py.Object.t -> ?struct_as_record:Py.Object.t -> ?verify_compressed_data_integrity:Py.Object.t -> ?uint16_codec:Py.Object.t -> ?simplify_cells:Py.Object.t -> mat_stream:Py.Object.t -> unit -> t
(**
Reader for Mat 5 mat files
Adds the following attribute to base class

uint16_codec - char codec to use for uint16 char arrays
    (defaults to system default codec)

Uses variable reader that has the following stardard interface (see
abstract class in ``miobase``::

   __init__(self, file_reader)
   read_header(self)
   array_from_header(self)

and added interface::

   set_stream(self, stream)
   read_full_tag(self)
*)

val end_of_stream : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val get_variables : ?variable_names:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
get variables from stream as dictionary

variable_names   - optional list of variable names to get

If variable_names is None, then get all variables in file
*)

val guess_byte_order : [> tag] Obj.t -> Py.Object.t
(**
Guess byte order.
Sets stream pointer to 0 
*)

val initialize_read : [> tag] Obj.t -> Py.Object.t
(**
Run when beginning read of variables

Sets up readers from parameters in `self`
*)

val list_variables : [> tag] Obj.t -> Py.Object.t
(**
list variables from stream 
*)

val read_file_header : [> tag] Obj.t -> Py.Object.t
(**
Read in mat 5 file header 
*)

val read_var_array : ?process:Py.Object.t -> header:Py.Object.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Read array, given `header`

Parameters
----------
header : header object
   object with fields defining variable header
process : {True, False} bool, optional
   If True, apply recursive post-processing during loading of
   array.

Returns
-------
arr : array
   array with post-processing applied or not according to
   `process`.
*)

val read_var_header : [> tag] Obj.t -> (Py.Object.t * int)
(**
Read header, return header, next position

Header has to define at least .name and .is_global

Parameters
----------
None

Returns
-------
header : object
   object that can be passed to self.read_var_array, and that
   has attributes .name and .is_global
next_position : int
   position in stream of next variable
*)

val set_matlab_compatible : [> tag] Obj.t -> Py.Object.t
(**
Sets options to return arrays as MATLAB loads them 
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MatFile5Writer : sig
type tag = [`MatFile5Writer]
type t = [`MatFile5Writer | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?do_compression:Py.Object.t -> ?unicode_strings:Py.Object.t -> ?global_vars:Py.Object.t -> ?long_field_names:Py.Object.t -> ?oned_as:Py.Object.t -> file_stream:Py.Object.t -> unit -> t
(**
Class for writing mat5 files 
*)

val put_variables : ?write_header:bool -> mdict:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Write variables in `mdict` to stream

Parameters
----------
mdict : mapping
   mapping with method ``items`` returns name, contents pairs where
   ``name`` which will appear in the matlab workspace in file load, and
   ``contents`` is something writeable to a matlab file, such as a NumPy
   array.
write_header : {None, True, False}, optional
   If True, then write the matlab file header before writing the
   variables. If None (the default) then write the file header
   if we are at position 0 in the stream. By setting False
   here, and setting the stream position to the end of the file,
   you can append variables to a matlab file
*)

val write_file_header : [> tag] Obj.t -> Py.Object.t
(**
None
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val contextmanager : Py.Object.t -> Py.Object.t
(**
@contextmanager decorator.

Typical usage:

    @contextmanager
    def some_generator(<arguments>):
        <setup>
        try:
            yield <value>
        finally:
            <cleanup>

This makes this:

    with some_generator(<arguments>) as <variable>:
        <body>

equivalent to this:

    <setup>
    try:
        <variable> = <value>
        <body>
    finally:
        <cleanup>
*)

val docfiller : Py.Object.t -> Py.Object.t
(**
None
*)

val get_matfile_version : Py.Object.t -> (Py.Object.t * int)
(**
Return major, minor tuple depending on apparent mat file type

Where:

 #. 0,x -> version 4 format mat files
 #. 1,x -> version 5 format mat files
 #. 2,x -> version 7.3 format mat files (HDF format)

Parameters
----------
fileobj : file_like
    object implementing seek() and read()

Returns
-------
major_version : {0, 1, 2}
    major MATLAB File format version
minor_version : int
    minor MATLAB file format version

Raises
------
MatReadError
    If the file is empty.
ValueError
    The matfile version is unknown.

Notes
-----
Has the side effect of setting the file read pointer to 0
*)

val loadmat : ?mdict:Py.Object.t -> ?appendmat:bool -> ?kwargs:(string * Py.Object.t) list -> file_name:string -> unit -> Py.Object.t
(**
Load MATLAB file.

Parameters
----------
file_name : str
   Name of the mat file (do not need .mat extension if
   appendmat==True). Can also pass open file-like object.
mdict : dict, optional
    Dictionary in which to insert matfile variables.
appendmat : bool, optional
   True to append the .mat extension to the end of the given
   filename, if not already present.
byte_order : str or None, optional
   None by default, implying byte order guessed from mat
   file. Otherwise can be one of ('native', '=', 'little', '<',
   'BIG', '>').
mat_dtype : bool, optional
   If True, return arrays in same dtype as would be loaded into
   MATLAB (instead of the dtype with which they are saved).
squeeze_me : bool, optional
   Whether to squeeze unit matrix dimensions or not.
chars_as_strings : bool, optional
   Whether to convert char arrays to string arrays.
matlab_compatible : bool, optional
   Returns matrices as would be loaded by MATLAB (implies
   squeeze_me=False, chars_as_strings=False, mat_dtype=True,
   struct_as_record=True).
struct_as_record : bool, optional
   Whether to load MATLAB structs as NumPy record arrays, or as
   old-style NumPy arrays with dtype=object. Setting this flag to
   False replicates the behavior of scipy version 0.7.x (returning
   NumPy object arrays). The default setting is True, because it
   allows easier round-trip load and save of MATLAB files.
verify_compressed_data_integrity : bool, optional
    Whether the length of compressed sequences in the MATLAB file
    should be checked, to ensure that they are not longer than we expect.
    It is advisable to enable this (the default) because overlong
    compressed sequences in MATLAB files generally indicate that the
    files have experienced some sort of corruption.
variable_names : None or sequence
    If None (the default) - read all variables in file. Otherwise,
    `variable_names` should be a sequence of strings, giving names of the
    MATLAB variables to read from the file. The reader will skip any
    variable with a name not in this sequence, possibly saving some read
    processing.
simplify_cells : False, optional
    If True, return a simplified dict structure (which is useful if the mat
    file contains cell arrays). Note that this only affects the structure
    of the result and not its contents (which is identical for both output
    structures). If True, this automatically sets `struct_as_record` to
    False and `squeeze_me` to True, which is required to simplify cells.

Returns
-------
mat_dict : dict
   dictionary with variable names as keys, and loaded matrices as
   values.

Notes
-----
v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.

You will need an HDF5 Python library to read MATLAB 7.3 format mat
files. Because SciPy does not supply one, we do not implement the
HDF5 / 7.3 interface here.

Examples
--------
>>> from os.path import dirname, join as pjoin
>>> import scipy.io as sio

Get the filename for an example .mat file from the tests/data directory.

>>> data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
>>> mat_fname = pjoin(data_dir, 'testdouble_7.4_GLNX86.mat')

Load the .mat file contents.

>>> mat_contents = sio.loadmat(mat_fname)

The result is a dictionary, one key/value pair for each variable:

>>> sorted(mat_contents.keys())
['__globals__', '__header__', '__version__', 'testdouble']
>>> mat_contents['testdouble']
array([[0.        , 0.78539816, 1.57079633, 2.35619449, 3.14159265,
        3.92699082, 4.71238898, 5.49778714, 6.28318531]])

By default SciPy reads MATLAB structs as structured NumPy arrays where the
dtype fields are of type `object` and the names correspond to the MATLAB
struct field names. This can be disabled by setting the optional argument
`struct_as_record=False`.

Get the filename for an example .mat file that contains a MATLAB struct
called `teststruct` and load the contents.

>>> matstruct_fname = pjoin(data_dir, 'teststruct_7.4_GLNX86.mat')
>>> matstruct_contents = sio.loadmat(matstruct_fname)
>>> teststruct = matstruct_contents['teststruct']
>>> teststruct.dtype
dtype([('stringfield', 'O'), ('doublefield', 'O'), ('complexfield', 'O')])

The size of the structured array is the size of the MATLAB struct, not the
number of elements in any particular field. The shape defaults to 2-D
unless the optional argument `squeeze_me=True`, in which case all length 1
dimensions are removed.

>>> teststruct.size
1
>>> teststruct.shape
(1, 1)

Get the 'stringfield' of the first element in the MATLAB struct.

>>> teststruct[0, 0]['stringfield']
array(['Rats live on no evil star.'],
  dtype='<U26')

Get the first element of the 'doublefield'.

>>> teststruct['doublefield'][0, 0]
array([[ 1.41421356,  2.71828183,  3.14159265]])

Load the MATLAB struct, squeezing out length 1 dimensions, and get the item
from the 'complexfield'.

>>> matstruct_squeezed = sio.loadmat(matstruct_fname, squeeze_me=True)
>>> matstruct_squeezed['teststruct'].shape
()
>>> matstruct_squeezed['teststruct']['complexfield'].shape
()
>>> matstruct_squeezed['teststruct']['complexfield'].item()
array([ 1.41421356+1.41421356j,  2.71828183+2.71828183j,
    3.14159265+3.14159265j])
*)

val mat_reader_factory : ?appendmat:bool -> ?kwargs:(string * Py.Object.t) list -> file_name:string -> unit -> (Py.Object.t * bool)
(**
Create reader for matlab .mat format files.

Parameters
----------
file_name : str
   Name of the mat file (do not need .mat extension if
   appendmat==True) Can also pass open file-like object.
appendmat : bool, optional
   True to append the .mat extension to the end of the given
   filename, if not already present.
byte_order : str or None, optional
   None by default, implying byte order guessed from mat
   file. Otherwise can be one of ('native', '=', 'little', '<',
   'BIG', '>').
mat_dtype : bool, optional
   If True, return arrays in same dtype as would be loaded into
   MATLAB (instead of the dtype with which they are saved).
squeeze_me : bool, optional
   Whether to squeeze unit matrix dimensions or not.
chars_as_strings : bool, optional
   Whether to convert char arrays to string arrays.
matlab_compatible : bool, optional
   Returns matrices as would be loaded by MATLAB (implies
   squeeze_me=False, chars_as_strings=False, mat_dtype=True,
   struct_as_record=True).
struct_as_record : bool, optional
   Whether to load MATLAB structs as NumPy record arrays, or as
   old-style NumPy arrays with dtype=object. Setting this flag to
   False replicates the behavior of SciPy version 0.7.x (returning
   numpy object arrays). The default setting is True, because it
   allows easier round-trip load and save of MATLAB files.

Returns
-------
matreader : MatFileReader object
   Initialized instance of MatFileReader class matching the mat file
   type detected in `filename`.
file_opened : bool
   Whether the file was opened by this routine.
*)

val savemat : ?appendmat:bool -> ?format:[`T4 | `T5] -> ?long_field_names:bool -> ?do_compression:bool -> ?oned_as:[`Row | `Column] -> file_name:[`S of string | `File_like_object of Py.Object.t] -> mdict:Py.Object.t -> unit -> Py.Object.t
(**
Save a dictionary of names and arrays into a MATLAB-style .mat file.

This saves the array objects in the given dictionary to a MATLAB-
style .mat file.

Parameters
----------
file_name : str or file-like object
    Name of the .mat file (.mat extension not needed if ``appendmat ==
    True``).
    Can also pass open file_like object.
mdict : dict
    Dictionary from which to save matfile variables.
appendmat : bool, optional
    True (the default) to append the .mat extension to the end of the
    given filename, if not already present.
format : {'5', '4'}, string, optional
    '5' (the default) for MATLAB 5 and up (to 7.2),
    '4' for MATLAB 4 .mat files.
long_field_names : bool, optional
    False (the default) - maximum field name length in a structure is
    31 characters which is the documented maximum length.
    True - maximum field name length in a structure is 63 characters
    which works for MATLAB 7.6+.
do_compression : bool, optional
    Whether or not to compress matrices on write. Default is False.
oned_as : {'row', 'column'}, optional
    If 'column', write 1-D NumPy arrays as column vectors.
    If 'row', write 1-D NumPy arrays as row vectors.

Examples
--------
>>> from scipy.io import savemat
>>> a = np.arange(20)
>>> mdic = {'a': a, 'label': 'experiment'}
>>> mdic
{'a': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19]),
'label': 'experiment'}
>>> savemat('matlab_matrix.mat', mdic)
*)

val whosmat : ?appendmat:bool -> ?kwargs:(string * Py.Object.t) list -> file_name:string -> unit -> Py.Object.t
(**
List variables inside a MATLAB file.

Parameters
----------
file_name : str
   Name of the mat file (do not need .mat extension if
   appendmat==True) Can also pass open file-like object.
appendmat : bool, optional
   True to append the .mat extension to the end of the given
   filename, if not already present.
byte_order : str or None, optional
   None by default, implying byte order guessed from mat
   file. Otherwise can be one of ('native', '=', 'little', '<',
   'BIG', '>').
mat_dtype : bool, optional
   If True, return arrays in same dtype as would be loaded into
   MATLAB (instead of the dtype with which they are saved).
squeeze_me : bool, optional
   Whether to squeeze unit matrix dimensions or not.
chars_as_strings : bool, optional
   Whether to convert char arrays to string arrays.
matlab_compatible : bool, optional
   Returns matrices as would be loaded by MATLAB (implies
   squeeze_me=False, chars_as_strings=False, mat_dtype=True,
   struct_as_record=True).
struct_as_record : bool, optional
   Whether to load MATLAB structs as NumPy record arrays, or as
   old-style NumPy arrays with dtype=object. Setting this flag to
   False replicates the behavior of SciPy version 0.7.x (returning
   numpy object arrays). The default setting is True, because it
   allows easier round-trip load and save of MATLAB files.

Returns
-------
variables : list of tuples
    A list of tuples, where each tuple holds the matrix name (a string),
    its shape (tuple of ints), and its data class (a string).
    Possible data classes are: int8, uint8, int16, uint16, int32, uint32,
    int64, uint64, single, double, cell, struct, object, char, sparse,
    function, opaque, logical, unknown.

Notes
-----
v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.

You will need an HDF5 python library to read matlab 7.3 format mat
files. Because SciPy does not supply one, we do not implement the
HDF5 / 7.3 interface here.

.. versionadded:: 0.12.0
*)


end

module Mio4 : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module MatFileReader : sig
type tag = [`MatFileReader]
type t = [`MatFileReader | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?byte_order:Py.Object.t -> ?mat_dtype:Py.Object.t -> ?squeeze_me:Py.Object.t -> ?chars_as_strings:Py.Object.t -> ?matlab_compatible:Py.Object.t -> ?struct_as_record:Py.Object.t -> ?verify_compressed_data_integrity:Py.Object.t -> ?simplify_cells:Py.Object.t -> mat_stream:Py.Object.t -> unit -> t
(**
Base object for reading mat files

To make this class functional, you will need to override the
following methods:

matrix_getter_factory   - gives object to fetch next matrix from stream
guess_byte_order        - guesses file byte order from file
*)

val end_of_stream : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val guess_byte_order : [> tag] Obj.t -> Py.Object.t
(**
As we do not know what file type we have, assume native 
*)

val set_matlab_compatible : [> tag] Obj.t -> Py.Object.t
(**
Sets options to return arrays as MATLAB loads them 
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module VarHeader4 : sig
type tag = [`VarHeader4]
type t = [`Object | `VarHeader4] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : name:Py.Object.t -> dtype:Py.Object.t -> mclass:Py.Object.t -> dims:Py.Object.t -> is_complex:Py.Object.t -> unit -> t
(**
None
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module VarReader4 : sig
type tag = [`VarReader4]
type t = [`Object | `VarReader4] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t -> t
(**
Class to read matlab 4 variables 
*)

val array_from_header : ?process:Py.Object.t -> hdr:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val read_char_array : hdr:Py.Object.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
latin-1 text matrix (char matrix) reader

Parameters
----------
hdr : ``VarHeader4`` instance

Returns
-------
arr : ndarray
    with dtype 'U1', shape given by `hdr` ``dims``
*)

val read_full_array : hdr:Py.Object.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Full (rather than sparse) matrix getter

Read matrix (array) can be real or complex

Parameters
----------
hdr : ``VarHeader4`` instance

Returns
-------
arr : ndarray
    complex array if ``hdr.is_complex`` is True, otherwise a real
    numeric array
*)

val read_header : [> tag] Obj.t -> Py.Object.t
(**
Read and return header for variable 
*)

val read_sparse_array : hdr:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Read and return sparse matrix type

Parameters
----------
hdr : ``VarHeader4`` instance

Returns
-------
arr : ``scipy.sparse.coo_matrix``
    with dtype ``float`` and shape read from the sparse matrix data

Notes
-----
MATLAB 4 real sparse arrays are saved in a N+1 by 3 array format, where
N is the number of non-zero values. Column 1 values [0:N] are the
(1-based) row indices of the each non-zero value, column 2 [0:N] are the
column indices, column 3 [0:N] are the (real) values. The last values
[-1,0:2] of the rows, column indices are shape[0] and shape[1]
respectively of the output matrix. The last value for the values column
is a padding 0. mrows and ncols values from the header give the shape of
the stored matrix, here [N+1, 3]. Complex data are saved as a 4 column
matrix, where the fourth column contains the imaginary component; the
last value is again 0. Complex sparse data do *not* have the header
``imagf`` field set to True; the fact that the data are complex is only
detectable because there are 4 storage columns.
*)

val read_sub_array : ?copy:bool -> hdr:Py.Object.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Mat4 read using header `hdr` dtype and dims

Parameters
----------
hdr : object
   object with attributes ``dtype``, ``dims``. dtype is assumed to be
   the correct endianness
copy : bool, optional
   copies array before return if True (default True)
   (buffer is usually read only)

Returns
-------
arr : ndarray
    of dtype given by `hdr` ``dtype`` and shape given by `hdr` ``dims``
*)

val shape_from_header : hdr:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Read the shape of the array described by the header.
The file position after this call is unspecified.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module VarWriter4 : sig
type tag = [`VarWriter4]
type t = [`Object | `VarWriter4] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t -> t
(**
None
*)

val write : arr:[>`Ndarray] Np.Obj.t -> name:string -> [> tag] Obj.t -> Py.Object.t
(**
Write matrix `arr`, with name `name`

Parameters
----------
arr : array_like
   array to write
name : str
   name in matlab workspace
*)

val write_bytes : arr:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val write_char : arr:Py.Object.t -> name:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val write_header : ?p:int -> ?t:int -> ?imagf:int -> name:string -> shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Write header for given data options

Parameters
----------
name : str
    name of variable
shape : sequence
   Shape of array as it will be read in matlab
P : int, optional
    code for mat4 data type, one of ``miDOUBLE, miSINGLE, miINT32,
    miINT16, miUINT16, miUINT8``
T : int, optional
    code for mat4 matrix class, one of ``mxFULL_CLASS, mxCHAR_CLASS,
    mxSPARSE_CLASS``
imagf : int, optional
    flag indicating complex
*)

val write_numeric : arr:Py.Object.t -> name:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val write_sparse : arr:Py.Object.t -> name:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Sparse matrices are 2-D

See docstring for VarReader4.read_sparse_array
*)

val write_string : s:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val arr_dtype_number : arr:Py.Object.t -> num:Py.Object.t -> unit -> Py.Object.t
(**
Return dtype for given number of items per element
*)

val arr_to_2d : ?oned_as:Py.Object.t -> arr:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Make ``arr`` exactly two dimensional

If `arr` has more than 2 dimensions, raise a ValueError

Parameters
----------
arr : array
oned_as : {'row', 'column'}, optional
   Whether to reshape 1-D vectors as row vectors or column vectors.
   See documentation for ``matdims`` for more detail

Returns
-------
arr2d : array
   2-D version of the array
*)

val arr_to_chars : Py.Object.t -> Py.Object.t
(**
Convert string array to char array 
*)

val asbytes : Py.Object.t -> Py.Object.t
(**
None
*)

val asstr : Py.Object.t -> Py.Object.t
(**
None
*)

val convert_dtypes : dtype_template:Py.Object.t -> order_code:string -> unit -> Py.Object.t
(**
Convert dtypes in mapping to given order

Parameters
----------
dtype_template : mapping
   mapping with values returning numpy dtype from ``np.dtype(val)``
order_code : str
   an order code suitable for using in ``dtype.newbyteorder()``

Returns
-------
dtypes : mapping
   mapping where values have been replaced by
   ``np.dtype(val).newbyteorder(order_code)``
*)

val docfiller : Py.Object.t -> Py.Object.t
(**
None
*)

val matdims : ?oned_as:[`Column | `Row] -> arr:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Determine equivalent MATLAB dimensions for given array

Parameters
----------
arr : ndarray
    Input array
oned_as : {'column', 'row'}, optional
    Whether 1-D arrays are returned as MATLAB row or column matrices.
    Default is 'column'.

Returns
-------
dims : tuple
    Shape tuple, in the form MATLAB expects it.

Notes
-----
We had to decide what shape a 1 dimensional array would be by
default. ``np.atleast_2d`` thinks it is a row vector. The
default for a vector in MATLAB (e.g., ``>> 1:12``) is a row vector.

Versions of scipy up to and including 0.11 resulted (accidentally)
in 1-D arrays being read as column vectors. For the moment, we
maintain the same tradition here.

Examples
--------
>>> matdims(np.array(1)) # NumPy scalar
(1, 1)
>>> matdims(np.array([1])) # 1-D array, 1 element
(1, 1)
>>> matdims(np.array([1,2])) # 1-D array, 2 elements
(2, 1)
>>> matdims(np.array([[2],[3]])) # 2-D array, column vector
(2, 1)
>>> matdims(np.array([[2,3]])) # 2-D array, row vector
(1, 2)
>>> matdims(np.array([[[2,3]]])) # 3-D array, rowish vector
(1, 1, 2)
>>> matdims(np.array([])) # empty 1-D array
(0, 0)
>>> matdims(np.array([[]])) # empty 2-D array
(0, 0)
>>> matdims(np.array([[[]]])) # empty 3-D array
(0, 0, 0)

Optional argument flips 1-D shape behavior.

>>> matdims(np.array([1,2]), 'row') # 1-D array, 2 elements
(1, 2)

The argument has to make sense though

>>> matdims(np.array([1,2]), 'bizarre')
Traceback (most recent call last):
   ...
ValueError: 1-D option 'bizarre' is strange
*)

val read_dtype : mat_stream:Py.Object.t -> a_dtype:Np.Dtype.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Generic get of byte stream data of known type

Parameters
----------
mat_stream : file_like object
    MATLAB (tm) mat file stream
a_dtype : dtype
    dtype of array to read. `a_dtype` is assumed to be correct
    endianness.

Returns
-------
arr : ndarray
    Array of dtype `a_dtype` read from stream.
*)

val reduce : ?initial:Py.Object.t -> function_:Py.Object.t -> sequence:Py.Object.t -> unit -> Py.Object.t
(**
reduce(function, sequence[, initial]) -> value

Apply a function of two arguments cumulatively to the items of a sequence,
from left to right, so as to reduce the sequence to a single value.
For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
((((1+2)+3)+4)+5).  If initial is present, it is placed before the items
of the sequence in the calculation, and serves as a default when the
sequence is empty.
*)


end

module Mio5 : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module BytesIO : sig
type tag = [`BytesIO]
type t = [`BytesIO | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?initial_bytes:Py.Object.t -> unit -> t
(**
Buffered I/O implementation using an in-memory bytes buffer.
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)

val close : [> tag] Obj.t -> Py.Object.t
(**
Disable all I/O operations.
*)

val detach : [> tag] Obj.t -> Py.Object.t
(**
Disconnect this buffer from its underlying raw stream and return it.

After the raw stream has been detached, the buffer is in an unusable
state.
*)

val fileno : [> tag] Obj.t -> Py.Object.t
(**
Returns underlying file descriptor if one exists.

OSError is raised if the IO object does not use a file descriptor.
*)

val flush : [> tag] Obj.t -> Py.Object.t
(**
Does nothing.
*)

val getbuffer : [> tag] Obj.t -> Py.Object.t
(**
Get a read-write view over the contents of the BytesIO object.
*)

val getvalue : [> tag] Obj.t -> Py.Object.t
(**
Retrieve the entire contents of the BytesIO object.
*)

val isatty : [> tag] Obj.t -> Py.Object.t
(**
Always returns False.

BytesIO objects are not connected to a TTY-like device.
*)

val read : ?size:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Read at most size bytes, returned as a bytes object.

If the size argument is negative, read until EOF is reached.
Return an empty bytes object at EOF.
*)

val read1 : ?size:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Read at most size bytes, returned as a bytes object.

If the size argument is negative or omitted, read until EOF is reached.
Return an empty bytes object at EOF.
*)

val readable : [> tag] Obj.t -> Py.Object.t
(**
Returns True if the IO object can be read.
*)

val readinto : buffer:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Read bytes into buffer.

Returns number of bytes read (0 for EOF), or None if the object
is set not to block and has no data to read.
*)

val readinto1 : buffer:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val readline : ?size:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Next line from the file, as a bytes object.

Retain newline.  A non-negative size argument limits the maximum
number of bytes to return (an incomplete line may be returned then).
Return an empty bytes object at EOF.
*)

val readlines : ?size:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
List of bytes objects, each a line from the file.

Call readline() repeatedly and return a list of the lines so read.
The optional size argument, if given, is an approximate bound on the
total number of bytes in the lines returned.
*)

val seek : ?whence:Py.Object.t -> pos:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Change stream position.

Seek to byte offset pos relative to position indicated by whence:
     0  Start of stream (the default).  pos should be >= 0;
     1  Current position - pos may be negative;
     2  End of stream - pos usually negative.
Returns the new absolute position.
*)

val seekable : [> tag] Obj.t -> Py.Object.t
(**
Returns True if the IO object can be seeked.
*)

val tell : [> tag] Obj.t -> Py.Object.t
(**
Current file position, an integer.
*)

val truncate : ?size:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Truncate the file to at most size bytes.

Size defaults to the current file position, as returned by tell().
The current file position is unchanged.  Returns the new size.
*)

val writable : [> tag] Obj.t -> Py.Object.t
(**
Returns True if the IO object can be written.
*)

val write : b:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Write bytes to file.

Return the number of bytes written.
*)

val writelines : lines:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Write lines to the file.

Note that newlines are not added.  lines can be any iterable object
producing bytes-like objects. This is equivalent to calling write() for
each element.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module EmptyStructMarker : sig
type tag = [`EmptyStructMarker]
type t = [`EmptyStructMarker | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Class to indicate presence of empty matlab struct on output 
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MatFileReader : sig
type tag = [`MatFileReader]
type t = [`MatFileReader | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?byte_order:Py.Object.t -> ?mat_dtype:Py.Object.t -> ?squeeze_me:Py.Object.t -> ?chars_as_strings:Py.Object.t -> ?matlab_compatible:Py.Object.t -> ?struct_as_record:Py.Object.t -> ?verify_compressed_data_integrity:Py.Object.t -> ?simplify_cells:Py.Object.t -> mat_stream:Py.Object.t -> unit -> t
(**
Base object for reading mat files

To make this class functional, you will need to override the
following methods:

matrix_getter_factory   - gives object to fetch next matrix from stream
guess_byte_order        - guesses file byte order from file
*)

val end_of_stream : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val guess_byte_order : [> tag] Obj.t -> Py.Object.t
(**
As we do not know what file type we have, assume native 
*)

val set_matlab_compatible : [> tag] Obj.t -> Py.Object.t
(**
Sets options to return arrays as MATLAB loads them 
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MatReadError : sig
type tag = [`MatReadError]
type t = [`BaseException | `MatReadError | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_exception : t -> [`BaseException] Obj.t
val with_traceback : tb:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MatReadWarning : sig
type tag = [`MatReadWarning]
type t = [`BaseException | `MatReadWarning | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_exception : t -> [`BaseException] Obj.t
val with_traceback : tb:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MatWriteError : sig
type tag = [`MatWriteError]
type t = [`BaseException | `MatWriteError | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_exception : t -> [`BaseException] Obj.t
val with_traceback : tb:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Exception.with_traceback(tb) --
set self.__traceback__ to tb and return self.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MatlabFunction : sig
type tag = [`MatlabFunction]
type t = [`ArrayLike | `MatlabFunction | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t -> t
(**
Subclass to signal this is a matlab function 
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)

val __setitem__ : key:Py.Object.t -> value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Set self[key] to value.
*)

val all : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.all(axis=None, out=None, keepdims=False)

Returns True if all elements evaluate to True.

Refer to `numpy.all` for full documentation.

See Also
--------
numpy.all : equivalent function
*)

val any : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.any(axis=None, out=None, keepdims=False)

Returns True if any of the elements of `a` evaluate to True.

Refer to `numpy.any` for full documentation.

See Also
--------
numpy.any : equivalent function
*)

val argmax : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argmax(axis=None, out=None)

Return indices of the maximum values along the given axis.

Refer to `numpy.argmax` for full documentation.

See Also
--------
numpy.argmax : equivalent function
*)

val argmin : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argmin(axis=None, out=None)

Return indices of the minimum values along the given axis of `a`.

Refer to `numpy.argmin` for detailed documentation.

See Also
--------
numpy.argmin : equivalent function
*)

val argpartition : ?axis:Py.Object.t -> ?kind:Py.Object.t -> ?order:Py.Object.t -> kth:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argpartition(kth, axis=-1, kind='introselect', order=None)

Returns the indices that would partition this array.

Refer to `numpy.argpartition` for full documentation.

.. versionadded:: 1.8.0

See Also
--------
numpy.argpartition : equivalent function
*)

val argsort : ?axis:Py.Object.t -> ?kind:Py.Object.t -> ?order:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argsort(axis=-1, kind=None, order=None)

Returns the indices that would sort this array.

Refer to `numpy.argsort` for full documentation.

See Also
--------
numpy.argsort : equivalent function
*)

val astype : ?order:[`C | `F | `A | `K] -> ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?subok:Py.Object.t -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)

Copy of the array, cast to a specified type.

Parameters
----------
dtype : str or dtype
    Typecode or data-type to which the array is cast.
order : {'C', 'F', 'A', 'K'}, optional
    Controls the memory layout order of the result.
    'C' means C order, 'F' means Fortran order, 'A'
    means 'F' order if all the arrays are Fortran contiguous,
    'C' order otherwise, and 'K' means as close to the
    order the array elements appear in memory as possible.
    Default is 'K'.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur. Defaults to 'unsafe'
    for backwards compatibility.

      * 'no' means the data types should not be cast at all.
      * 'equiv' means only byte-order changes are allowed.
      * 'safe' means only casts which can preserve values are allowed.
      * 'same_kind' means only safe casts or casts within a kind,
        like float64 to float32, are allowed.
      * 'unsafe' means any data conversions may be done.
subok : bool, optional
    If True, then sub-classes will be passed-through (default), otherwise
    the returned array will be forced to be a base-class array.
copy : bool, optional
    By default, astype always returns a newly allocated array. If this
    is set to false, and the `dtype`, `order`, and `subok`
    requirements are satisfied, the input array is returned instead
    of a copy.

Returns
-------
arr_t : ndarray
    Unless `copy` is False and the other conditions for returning the input
    array are satisfied (see description for `copy` input parameter), `arr_t`
    is a new array of the same shape as the input array, with dtype, order
    given by `dtype`, `order`.

Notes
-----
.. versionchanged:: 1.17.0
   Casting between a simple data type and a structured one is possible only
   for 'unsafe' casting.  Casting to multiple fields is allowed, but
   casting from multiple fields is not.

.. versionchanged:: 1.9.0
   Casting from numeric to string types in 'safe' casting mode requires
   that the string dtype length is long enough to store the max
   integer/float value converted.

Raises
------
ComplexWarning
    When casting from complex to float or int. To avoid this,
    one should use ``a.real.astype(t)``.

Examples
--------
>>> x = np.array([1, 2, 2.5])
>>> x
array([1. ,  2. ,  2.5])

>>> x.astype(int)
array([1, 2, 2])
*)

val byteswap : ?inplace:bool -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.byteswap(inplace=False)

Swap the bytes of the array elements

Toggle between low-endian and big-endian data representation by
returning a byteswapped array, optionally swapped in-place.
Arrays of byte-strings are not swapped. The real and imaginary
parts of a complex number are swapped individually.

Parameters
----------
inplace : bool, optional
    If ``True``, swap bytes in-place, default is ``False``.

Returns
-------
out : ndarray
    The byteswapped array. If `inplace` is ``True``, this is
    a view to self.

Examples
--------
>>> A = np.array([1, 256, 8755], dtype=np.int16)
>>> list(map(hex, A))
['0x1', '0x100', '0x2233']
>>> A.byteswap(inplace=True)
array([  256,     1, 13090], dtype=int16)
>>> list(map(hex, A))
['0x100', '0x1', '0x3322']

Arrays of byte-strings are not swapped

>>> A = np.array([b'ceg', b'fac'])
>>> A.byteswap()
array([b'ceg', b'fac'], dtype='|S3')

``A.newbyteorder().byteswap()`` produces an array with the same values
  but different representation in memory

>>> A = np.array([1, 2, 3])
>>> A.view(np.uint8)
array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
       0, 0], dtype=uint8)
>>> A.newbyteorder().byteswap(inplace=True)
array([1, 2, 3])
>>> A.view(np.uint8)
array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
       0, 3], dtype=uint8)
*)

val choose : ?out:Py.Object.t -> ?mode:Py.Object.t -> choices:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.choose(choices, out=None, mode='raise')

Use an index array to construct a new array from a set of choices.

Refer to `numpy.choose` for full documentation.

See Also
--------
numpy.choose : equivalent function
*)

val clip : ?min:Py.Object.t -> ?max:Py.Object.t -> ?out:Py.Object.t -> ?kwargs:(string * Py.Object.t) list -> [> tag] Obj.t -> Py.Object.t
(**
a.clip(min=None, max=None, out=None, **kwargs)

Return an array whose values are limited to ``[min, max]``.
One of max or min must be given.

Refer to `numpy.clip` for full documentation.

See Also
--------
numpy.clip : equivalent function
*)

val compress : ?axis:Py.Object.t -> ?out:Py.Object.t -> condition:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.compress(condition, axis=None, out=None)

Return selected slices of this array along given axis.

Refer to `numpy.compress` for full documentation.

See Also
--------
numpy.compress : equivalent function
*)

val conj : [> tag] Obj.t -> Py.Object.t
(**
a.conj()

Complex-conjugate all elements.

Refer to `numpy.conjugate` for full documentation.

See Also
--------
numpy.conjugate : equivalent function
*)

val conjugate : [> tag] Obj.t -> Py.Object.t
(**
a.conjugate()

Return the complex conjugate, element-wise.

Refer to `numpy.conjugate` for full documentation.

See Also
--------
numpy.conjugate : equivalent function
*)

val copy : ?order:[`C | `F | `A | `K] -> [> tag] Obj.t -> Py.Object.t
(**
a.copy(order='C')

Return a copy of the array.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    Controls the memory layout of the copy. 'C' means C-order,
    'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
    'C' otherwise. 'K' means match the layout of `a` as closely
    as possible. (Note that this function and :func:`numpy.copy` are very
    similar, but have different default values for their order=
    arguments.)

See also
--------
numpy.copy
numpy.copyto

Examples
--------
>>> x = np.array([[1,2,3],[4,5,6]], order='F')

>>> y = x.copy()

>>> x.fill(0)

>>> x
array([[0, 0, 0],
       [0, 0, 0]])

>>> y
array([[1, 2, 3],
       [4, 5, 6]])

>>> y.flags['C_CONTIGUOUS']
True
*)

val cumprod : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.cumprod(axis=None, dtype=None, out=None)

Return the cumulative product of the elements along the given axis.

Refer to `numpy.cumprod` for full documentation.

See Also
--------
numpy.cumprod : equivalent function
*)

val cumsum : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.cumsum(axis=None, dtype=None, out=None)

Return the cumulative sum of the elements along the given axis.

Refer to `numpy.cumsum` for full documentation.

See Also
--------
numpy.cumsum : equivalent function
*)

val diagonal : ?offset:Py.Object.t -> ?axis1:Py.Object.t -> ?axis2:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.diagonal(offset=0, axis1=0, axis2=1)

Return specified diagonals. In NumPy 1.9 the returned array is a
read-only view instead of a copy as in previous NumPy versions.  In
a future version the read-only restriction will be removed.

Refer to :func:`numpy.diagonal` for full documentation.

See Also
--------
numpy.diagonal : equivalent function
*)

val dot : ?out:Py.Object.t -> b:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.dot(b, out=None)

Dot product of two arrays.

Refer to `numpy.dot` for full documentation.

See Also
--------
numpy.dot : equivalent function

Examples
--------
>>> a = np.eye(2)
>>> b = np.ones((2, 2)) * 2
>>> a.dot(b)
array([[2.,  2.],
       [2.,  2.]])

This array method can be conveniently chained:

>>> a.dot(b).dot(b)
array([[8.,  8.],
       [8.,  8.]])
*)

val dump : file:[`S of string | `Path of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.dump(file)

Dump a pickle of the array to the specified file.
The array can be read back with pickle.load or numpy.load.

Parameters
----------
file : str or Path
    A string naming the dump file.

    .. versionchanged:: 1.17.0
        `pathlib.Path` objects are now accepted.
*)

val dumps : [> tag] Obj.t -> Py.Object.t
(**
a.dumps()

Returns the pickle of the array as a string.
pickle.loads or numpy.loads will convert the string back to an array.

Parameters
----------
None
*)

val fill : value:[`F of float | `I of int | `Bool of bool | `S of string] -> [> tag] Obj.t -> Py.Object.t
(**
a.fill(value)

Fill the array with a scalar value.

Parameters
----------
value : scalar
    All elements of `a` will be assigned this value.

Examples
--------
>>> a = np.array([1, 2])
>>> a.fill(0)
>>> a
array([0, 0])
>>> a = np.empty(2)
>>> a.fill(1)
>>> a
array([1.,  1.])
*)

val flatten : ?order:[`C | `F | `A | `K] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.flatten(order='C')

Return a copy of the array collapsed into one dimension.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    'C' means to flatten in row-major (C-style) order.
    'F' means to flatten in column-major (Fortran-
    style) order. 'A' means to flatten in column-major
    order if `a` is Fortran *contiguous* in memory,
    row-major order otherwise. 'K' means to flatten
    `a` in the order the elements occur in memory.
    The default is 'C'.

Returns
-------
y : ndarray
    A copy of the input array, flattened to one dimension.

See Also
--------
ravel : Return a flattened array.
flat : A 1-D flat iterator over the array.

Examples
--------
>>> a = np.array([[1,2], [3,4]])
>>> a.flatten()
array([1, 2, 3, 4])
>>> a.flatten('F')
array([1, 3, 2, 4])
*)

val getfield : ?offset:int -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.getfield(dtype, offset=0)

Returns a field of the given array as a certain type.

A field is a view of the array data with a given data-type. The values in
the view are determined by the given type and the offset into the current
array in bytes. The offset needs to be such that the view dtype fits in the
array dtype; for example an array of dtype complex128 has 16-byte elements.
If taking a view with a 32-bit integer (4 bytes), the offset needs to be
between 0 and 12 bytes.

Parameters
----------
dtype : str or dtype
    The data type of the view. The dtype size of the view can not be larger
    than that of the array itself.
offset : int
    Number of bytes to skip before beginning the element view.

Examples
--------
>>> x = np.diag([1.+1.j]*2)
>>> x[1, 1] = 2 + 4.j
>>> x
array([[1.+1.j,  0.+0.j],
       [0.+0.j,  2.+4.j]])
>>> x.getfield(np.float64)
array([[1.,  0.],
       [0.,  2.]])

By choosing an offset of 8 bytes we can select the complex part of the
array for our view:

>>> x.getfield(np.float64, offset=8)
array([[1.,  0.],
       [0.,  4.]])
*)

val item : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
a.item( *args)

Copy an element of an array to a standard Python scalar and return it.

Parameters
----------
\*args : Arguments (variable number and type)

    * none: in this case, the method only works for arrays
      with one element (`a.size == 1`), which element is
      copied into a standard Python scalar object and returned.

    * int_type: this argument is interpreted as a flat index into
      the array, specifying which element to copy and return.

    * tuple of int_types: functions as does a single int_type argument,
      except that the argument is interpreted as an nd-index into the
      array.

Returns
-------
z : Standard Python scalar object
    A copy of the specified element of the array as a suitable
    Python scalar

Notes
-----
When the data type of `a` is longdouble or clongdouble, item() returns
a scalar array object because there is no available Python scalar that
would not lose information. Void arrays return a buffer object for item(),
unless fields are defined, in which case a tuple is returned.

`item` is very similar to a[args], except, instead of an array scalar,
a standard Python scalar is returned. This can be useful for speeding up
access to elements of the array and doing arithmetic on elements of the
array using Python's optimized math.

Examples
--------
>>> np.random.seed(123)
>>> x = np.random.randint(9, size=(3, 3))
>>> x
array([[2, 2, 6],
       [1, 3, 6],
       [1, 0, 1]])
>>> x.item(3)
1
>>> x.item(7)
0
>>> x.item((0, 1))
2
>>> x.item((2, 2))
1
*)

val itemset : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
a.itemset( *args)

Insert scalar into an array (scalar is cast to array's dtype, if possible)

There must be at least 1 argument, and define the last argument
as *item*.  Then, ``a.itemset( *args)`` is equivalent to but faster
than ``a[args] = item``.  The item should be a scalar value and `args`
must select a single item in the array `a`.

Parameters
----------
\*args : Arguments
    If one argument: a scalar, only used in case `a` is of size 1.
    If two arguments: the last argument is the value to be set
    and must be a scalar, the first argument specifies a single array
    element location. It is either an int or a tuple.

Notes
-----
Compared to indexing syntax, `itemset` provides some speed increase
for placing a scalar into a particular location in an `ndarray`,
if you must do this.  However, generally this is discouraged:
among other problems, it complicates the appearance of the code.
Also, when using `itemset` (and `item`) inside a loop, be sure
to assign the methods to a local variable to avoid the attribute
look-up at each loop iteration.

Examples
--------
>>> np.random.seed(123)
>>> x = np.random.randint(9, size=(3, 3))
>>> x
array([[2, 2, 6],
       [1, 3, 6],
       [1, 0, 1]])
>>> x.itemset(4, 0)
>>> x.itemset((2, 2), 9)
>>> x
array([[2, 2, 6],
       [1, 0, 6],
       [1, 0, 9]])
*)

val max : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> ?initial:Py.Object.t -> ?where:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

Return the maximum along a given axis.

Refer to `numpy.amax` for full documentation.

See Also
--------
numpy.amax : equivalent function
*)

val mean : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.mean(axis=None, dtype=None, out=None, keepdims=False)

Returns the average of the array elements along given axis.

Refer to `numpy.mean` for full documentation.

See Also
--------
numpy.mean : equivalent function
*)

val min : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> ?initial:Py.Object.t -> ?where:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

Return the minimum along a given axis.

Refer to `numpy.amin` for full documentation.

See Also
--------
numpy.amin : equivalent function
*)

val newbyteorder : ?new_order:string -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
arr.newbyteorder(new_order='S')

Return the array with the same data viewed with a different byte order.

Equivalent to::

    arr.view(arr.dtype.newbytorder(new_order))

Changes are also made in all fields and sub-arrays of the array data
type.



Parameters
----------
new_order : string, optional
    Byte order to force; a value from the byte order specifications
    below. `new_order` codes can be any of:

    * 'S' - swap dtype from current to opposite endian
    * {'<', 'L'} - little endian
    * {'>', 'B'} - big endian
    * {'=', 'N'} - native order
    * {'|', 'I'} - ignore (no change to byte order)

    The default value ('S') results in swapping the current
    byte order. The code does a case-insensitive check on the first
    letter of `new_order` for the alternatives above.  For example,
    any of 'B' or 'b' or 'biggish' are valid to specify big-endian.


Returns
-------
new_arr : array
    New array object with the dtype reflecting given change to the
    byte order.
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
(**
a.nonzero()

Return the indices of the elements that are non-zero.

Refer to `numpy.nonzero` for full documentation.

See Also
--------
numpy.nonzero : equivalent function
*)

val partition : ?axis:int -> ?kind:[`Introselect] -> ?order:[`S of string | `StringList of string list] -> kth:[`I of int | `Is of int list] -> [> tag] Obj.t -> Py.Object.t
(**
a.partition(kth, axis=-1, kind='introselect', order=None)

Rearranges the elements in the array in such a way that the value of the
element in kth position is in the position it would be in a sorted array.
All elements smaller than the kth element are moved before this element and
all equal or greater are moved behind it. The ordering of the elements in
the two partitions is undefined.

.. versionadded:: 1.8.0

Parameters
----------
kth : int or sequence of ints
    Element index to partition by. The kth element value will be in its
    final sorted position and all smaller elements will be moved before it
    and all equal or greater elements behind it.
    The order of all elements in the partitions is undefined.
    If provided with a sequence of kth it will partition all elements
    indexed by kth of them into their sorted position at once.
axis : int, optional
    Axis along which to sort. Default is -1, which means sort along the
    last axis.
kind : {'introselect'}, optional
    Selection algorithm. Default is 'introselect'.
order : str or list of str, optional
    When `a` is an array with fields defined, this argument specifies
    which fields to compare first, second, etc. A single field can
    be specified as a string, and not all fields need to be specified,
    but unspecified fields will still be used, in the order in which
    they come up in the dtype, to break ties.

See Also
--------
numpy.partition : Return a parititioned copy of an array.
argpartition : Indirect partition.
sort : Full sort.

Notes
-----
See ``np.partition`` for notes on the different algorithms.

Examples
--------
>>> a = np.array([3, 4, 2, 1])
>>> a.partition(3)
>>> a
array([2, 1, 3, 4])

>>> a.partition((1, 3))
>>> a
array([1, 2, 3, 4])
*)

val prod : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> ?initial:Py.Object.t -> ?where:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.prod(axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True)

Return the product of the array elements over the given axis

Refer to `numpy.prod` for full documentation.

See Also
--------
numpy.prod : equivalent function
*)

val ptp : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.ptp(axis=None, out=None, keepdims=False)

Peak to peak (maximum - minimum) value along a given axis.

Refer to `numpy.ptp` for full documentation.

See Also
--------
numpy.ptp : equivalent function
*)

val put : ?mode:Py.Object.t -> indices:Py.Object.t -> values:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.put(indices, values, mode='raise')

Set ``a.flat[n] = values[n]`` for all `n` in indices.

Refer to `numpy.put` for full documentation.

See Also
--------
numpy.put : equivalent function
*)

val ravel : ?order:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.ravel([order])

Return a flattened array.

Refer to `numpy.ravel` for full documentation.

See Also
--------
numpy.ravel : equivalent function

ndarray.flat : a flat iterator on the array.
*)

val repeat : ?axis:Py.Object.t -> repeats:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.repeat(repeats, axis=None)

Repeat elements of an array.

Refer to `numpy.repeat` for full documentation.

See Also
--------
numpy.repeat : equivalent function
*)

val reshape : ?order:Py.Object.t -> shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.reshape(shape, order='C')

Returns an array containing the same data with a new shape.

Refer to `numpy.reshape` for full documentation.

See Also
--------
numpy.reshape : equivalent function

Notes
-----
Unlike the free function `numpy.reshape`, this method on `ndarray` allows
the elements of the shape parameter to be passed in as separate arguments.
For example, ``a.reshape(10, 11)`` is equivalent to
``a.reshape((10, 11))``.
*)

val resize : ?refcheck:bool -> new_shape:[`T_n_ints of Py.Object.t | `TupleOfInts of int list] -> [> tag] Obj.t -> Py.Object.t
(**
a.resize(new_shape, refcheck=True)

Change shape and size of array in-place.

Parameters
----------
new_shape : tuple of ints, or `n` ints
    Shape of resized array.
refcheck : bool, optional
    If False, reference count will not be checked. Default is True.

Returns
-------
None

Raises
------
ValueError
    If `a` does not own its own data or references or views to it exist,
    and the data memory must be changed.
    PyPy only: will always raise if the data memory must be changed, since
    there is no reliable way to determine if references or views to it
    exist.

SystemError
    If the `order` keyword argument is specified. This behaviour is a
    bug in NumPy.

See Also
--------
resize : Return a new array with the specified shape.

Notes
-----
This reallocates space for the data area if necessary.

Only contiguous arrays (data elements consecutive in memory) can be
resized.

The purpose of the reference count check is to make sure you
do not use this array as a buffer for another Python object and then
reallocate the memory. However, reference counts can increase in
other ways so if you are sure that you have not shared the memory
for this array with another Python object, then you may safely set
`refcheck` to False.

Examples
--------
Shrinking an array: array is flattened (in the order that the data are
stored in memory), resized, and reshaped:

>>> a = np.array([[0, 1], [2, 3]], order='C')
>>> a.resize((2, 1))
>>> a
array([[0],
       [1]])

>>> a = np.array([[0, 1], [2, 3]], order='F')
>>> a.resize((2, 1))
>>> a
array([[0],
       [2]])

Enlarging an array: as above, but missing entries are filled with zeros:

>>> b = np.array([[0, 1], [2, 3]])
>>> b.resize(2, 3) # new_shape parameter doesn't have to be a tuple
>>> b
array([[0, 1, 2],
       [3, 0, 0]])

Referencing an array prevents resizing...

>>> c = a
>>> a.resize((1, 1))
Traceback (most recent call last):
...
ValueError: cannot resize an array that references or is referenced ...

Unless `refcheck` is False:

>>> a.resize((1, 1), refcheck=False)
>>> a
array([[0]])
>>> c
array([[0]])
*)

val round : ?decimals:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.round(decimals=0, out=None)

Return `a` with each element rounded to the given number of decimals.

Refer to `numpy.around` for full documentation.

See Also
--------
numpy.around : equivalent function
*)

val searchsorted : ?side:Py.Object.t -> ?sorter:Py.Object.t -> v:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.searchsorted(v, side='left', sorter=None)

Find indices where elements of v should be inserted in a to maintain order.

For full documentation, see `numpy.searchsorted`

See Also
--------
numpy.searchsorted : equivalent function
*)

val setfield : ?offset:int -> val_:Py.Object.t -> dtype:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.setfield(val, dtype, offset=0)

Put a value into a specified place in a field defined by a data-type.

Place `val` into `a`'s field defined by `dtype` and beginning `offset`
bytes into the field.

Parameters
----------
val : object
    Value to be placed in field.
dtype : dtype object
    Data-type of the field in which to place `val`.
offset : int, optional
    The number of bytes into the field at which to place `val`.

Returns
-------
None

See Also
--------
getfield

Examples
--------
>>> x = np.eye(3)
>>> x.getfield(np.float64)
array([[1.,  0.,  0.],
       [0.,  1.,  0.],
       [0.,  0.,  1.]])
>>> x.setfield(3, np.int32)
>>> x.getfield(np.int32)
array([[3, 3, 3],
       [3, 3, 3],
       [3, 3, 3]], dtype=int32)
>>> x
array([[1.0e+000, 1.5e-323, 1.5e-323],
       [1.5e-323, 1.0e+000, 1.5e-323],
       [1.5e-323, 1.5e-323, 1.0e+000]])
>>> x.setfield(np.eye(3), np.int32)
>>> x
array([[1.,  0.,  0.],
       [0.,  1.,  0.],
       [0.,  0.,  1.]])
*)

val setflags : ?write:bool -> ?align:bool -> ?uic:bool -> [> tag] Obj.t -> Py.Object.t
(**
a.setflags(write=None, align=None, uic=None)

Set array flags WRITEABLE, ALIGNED, (WRITEBACKIFCOPY and UPDATEIFCOPY),
respectively.

These Boolean-valued flags affect how numpy interprets the memory
area used by `a` (see Notes below). The ALIGNED flag can only
be set to True if the data is actually aligned according to the type.
The WRITEBACKIFCOPY and (deprecated) UPDATEIFCOPY flags can never be set
to True. The flag WRITEABLE can only be set to True if the array owns its
own memory, or the ultimate owner of the memory exposes a writeable buffer
interface, or is a string. (The exception for string is made so that
unpickling can be done without copying memory.)

Parameters
----------
write : bool, optional
    Describes whether or not `a` can be written to.
align : bool, optional
    Describes whether or not `a` is aligned properly for its type.
uic : bool, optional
    Describes whether or not `a` is a copy of another 'base' array.

Notes
-----
Array flags provide information about how the memory area used
for the array is to be interpreted. There are 7 Boolean flags
in use, only four of which can be changed by the user:
WRITEBACKIFCOPY, UPDATEIFCOPY, WRITEABLE, and ALIGNED.

WRITEABLE (W) the data area can be written to;

ALIGNED (A) the data and strides are aligned appropriately for the hardware
(as determined by the compiler);

UPDATEIFCOPY (U) (deprecated), replaced by WRITEBACKIFCOPY;

WRITEBACKIFCOPY (X) this array is a copy of some other array (referenced
by .base). When the C-API function PyArray_ResolveWritebackIfCopy is
called, the base array will be updated with the contents of this array.

All flags can be accessed using the single (upper case) letter as well
as the full name.

Examples
--------
>>> y = np.array([[3, 1, 7],
...               [2, 0, 0],
...               [8, 5, 9]])
>>> y
array([[3, 1, 7],
       [2, 0, 0],
       [8, 5, 9]])
>>> y.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : True
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
>>> y.setflags(write=0, align=0)
>>> y.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : True
  WRITEABLE : False
  ALIGNED : False
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
>>> y.setflags(uic=1)
Traceback (most recent call last):
  File '<stdin>', line 1, in <module>
ValueError: cannot set WRITEBACKIFCOPY flag to True
*)

val sort : ?axis:int -> ?kind:[`Quicksort | `Heapsort | `Stable | `Mergesort] -> ?order:[`S of string | `StringList of string list] -> [> tag] Obj.t -> Py.Object.t
(**
a.sort(axis=-1, kind=None, order=None)

Sort an array in-place. Refer to `numpy.sort` for full documentation.

Parameters
----------
axis : int, optional
    Axis along which to sort. Default is -1, which means sort along the
    last axis.
kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
    Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
    and 'mergesort' use timsort under the covers and, in general, the
    actual implementation will vary with datatype. The 'mergesort' option
    is retained for backwards compatibility.

    .. versionchanged:: 1.15.0.
       The 'stable' option was added.

order : str or list of str, optional
    When `a` is an array with fields defined, this argument specifies
    which fields to compare first, second, etc.  A single field can
    be specified as a string, and not all fields need be specified,
    but unspecified fields will still be used, in the order in which
    they come up in the dtype, to break ties.

See Also
--------
numpy.sort : Return a sorted copy of an array.
numpy.argsort : Indirect sort.
numpy.lexsort : Indirect stable sort on multiple keys.
numpy.searchsorted : Find elements in sorted array.
numpy.partition: Partial sort.

Notes
-----
See `numpy.sort` for notes on the different sorting algorithms.

Examples
--------
>>> a = np.array([[1,4], [3,1]])
>>> a.sort(axis=1)
>>> a
array([[1, 4],
       [1, 3]])
>>> a.sort(axis=0)
>>> a
array([[1, 3],
       [1, 4]])

Use the `order` keyword to specify a field to use when sorting a
structured array:

>>> a = np.array([('a', 2), ('c', 1)], dtype=[('x', 'S1'), ('y', int)])
>>> a.sort(order='y')
>>> a
array([(b'c', 1), (b'a', 2)],
      dtype=[('x', 'S1'), ('y', '<i8')])
*)

val squeeze : ?axis:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.squeeze(axis=None)

Remove single-dimensional entries from the shape of `a`.

Refer to `numpy.squeeze` for full documentation.

See Also
--------
numpy.squeeze : equivalent function
*)

val std : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?ddof:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.std(axis=None, dtype=None, out=None, ddof=0, keepdims=False)

Returns the standard deviation of the array elements along given axis.

Refer to `numpy.std` for full documentation.

See Also
--------
numpy.std : equivalent function
*)

val sum : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> ?initial:Py.Object.t -> ?where:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.sum(axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True)

Return the sum of the array elements over the given axis.

Refer to `numpy.sum` for full documentation.

See Also
--------
numpy.sum : equivalent function
*)

val swapaxes : axis1:Py.Object.t -> axis2:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.swapaxes(axis1, axis2)

Return a view of the array with `axis1` and `axis2` interchanged.

Refer to `numpy.swapaxes` for full documentation.

See Also
--------
numpy.swapaxes : equivalent function
*)

val take : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?mode:Py.Object.t -> indices:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.take(indices, axis=None, out=None, mode='raise')

Return an array formed from the elements of `a` at the given indices.

Refer to `numpy.take` for full documentation.

See Also
--------
numpy.take : equivalent function
*)

val tobytes : ?order:[`C | `F | `None] -> [> tag] Obj.t -> Py.Object.t
(**
a.tobytes(order='C')

Construct Python bytes containing the raw data bytes in the array.

Constructs Python bytes showing a copy of the raw contents of
data memory. The bytes object can be produced in either 'C' or 'Fortran',
or 'Any' order (the default is 'C'-order). 'Any' order means C-order
unless the F_CONTIGUOUS flag in the array is set, in which case it
means 'Fortran' order.

.. versionadded:: 1.9.0

Parameters
----------
order : {'C', 'F', None}, optional
    Order of the data for multidimensional arrays:
    C, Fortran, or the same as for the original array.

Returns
-------
s : bytes
    Python bytes exhibiting a copy of `a`'s raw data.

Examples
--------
>>> x = np.array([[0, 1], [2, 3]], dtype='<u2')
>>> x.tobytes()
b'\x00\x00\x01\x00\x02\x00\x03\x00'
>>> x.tobytes('C') == x.tobytes()
True
>>> x.tobytes('F')
b'\x00\x00\x02\x00\x01\x00\x03\x00'
*)

val tofile : ?sep:string -> ?format:string -> fid:[`S of string | `PyObject of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.tofile(fid, sep='', format='%s')

Write array to a file as text or binary (default).

Data is always written in 'C' order, independent of the order of `a`.
The data produced by this method can be recovered using the function
fromfile().

Parameters
----------
fid : file or str or Path
    An open file object, or a string containing a filename.

    .. versionchanged:: 1.17.0
        `pathlib.Path` objects are now accepted.

sep : str
    Separator between array items for text output.
    If '' (empty), a binary file is written, equivalent to
    ``file.write(a.tobytes())``.
format : str
    Format string for text file output.
    Each entry in the array is formatted to text by first converting
    it to the closest Python type, and then using 'format' % item.

Notes
-----
This is a convenience function for quick storage of array data.
Information on endianness and precision is lost, so this method is not a
good choice for files intended to archive data or transport data between
machines with different endianness. Some of these problems can be overcome
by outputting the data as text files, at the expense of speed and file
size.

When fid is a file object, array contents are directly written to the
file, bypassing the file object's ``write`` method. As a result, tofile
cannot be used with files objects supporting compression (e.g., GzipFile)
or file-like objects that do not support ``fileno()`` (e.g., BytesIO).
*)

val tolist : [> tag] Obj.t -> Py.Object.t
(**
a.tolist()

Return the array as an ``a.ndim``-levels deep nested list of Python scalars.

Return a copy of the array data as a (nested) Python list.
Data items are converted to the nearest compatible builtin Python type, via
the `~numpy.ndarray.item` function.

If ``a.ndim`` is 0, then since the depth of the nested list is 0, it will
not be a list at all, but a simple Python scalar.

Parameters
----------
none

Returns
-------
y : object, or list of object, or list of list of object, or ...
    The possibly nested list of array elements.

Notes
-----
The array may be recreated via ``a = np.array(a.tolist())``, although this
may sometimes lose precision.

Examples
--------
For a 1D array, ``a.tolist()`` is almost the same as ``list(a)``,
except that ``tolist`` changes numpy scalars to Python scalars:

>>> a = np.uint32([1, 2])
>>> a_list = list(a)
>>> a_list
[1, 2]
>>> type(a_list[0])
<class 'numpy.uint32'>
>>> a_tolist = a.tolist()
>>> a_tolist
[1, 2]
>>> type(a_tolist[0])
<class 'int'>

Additionally, for a 2D array, ``tolist`` applies recursively:

>>> a = np.array([[1, 2], [3, 4]])
>>> list(a)
[array([1, 2]), array([3, 4])]
>>> a.tolist()
[[1, 2], [3, 4]]

The base case for this recursion is a 0D array:

>>> a = np.array(1)
>>> list(a)
Traceback (most recent call last):
  ...
TypeError: iteration over a 0-d array
>>> a.tolist()
1
*)

val tostring : ?order:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.tostring(order='C')

A compatibility alias for `tobytes`, with exactly the same behavior.

Despite its name, it returns `bytes` not `str`\ s.

.. deprecated:: 1.19.0
*)

val trace : ?offset:Py.Object.t -> ?axis1:Py.Object.t -> ?axis2:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)

Return the sum along diagonals of the array.

Refer to `numpy.trace` for full documentation.

See Also
--------
numpy.trace : equivalent function
*)

val transpose : Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.transpose( *axes)

Returns a view of the array with axes transposed.

For a 1-D array this has no effect, as a transposed vector is simply the
same vector. To convert a 1-D array into a 2D column vector, an additional
dimension must be added. `np.atleast2d(a).T` achieves this, as does
`a[:, np.newaxis]`.
For a 2-D array, this is a standard matrix transpose.
For an n-D array, if axes are given, their order indicates how the
axes are permuted (see Examples). If axes are not provided and
``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

Parameters
----------
axes : None, tuple of ints, or `n` ints

 * None or no argument: reverses the order of the axes.

 * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
   `i`-th axis becomes `a.transpose()`'s `j`-th axis.

 * `n` ints: same as an n-tuple of the same ints (this form is
   intended simply as a 'convenience' alternative to the tuple form)

Returns
-------
out : ndarray
    View of `a`, with axes suitably permuted.

See Also
--------
ndarray.T : Array property returning the array transposed.
ndarray.reshape : Give a new shape to an array without changing its data.

Examples
--------
>>> a = np.array([[1, 2], [3, 4]])
>>> a
array([[1, 2],
       [3, 4]])
>>> a.transpose()
array([[1, 3],
       [2, 4]])
>>> a.transpose((1, 0))
array([[1, 3],
       [2, 4]])
>>> a.transpose(1, 0)
array([[1, 3],
       [2, 4]])
*)

val var : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?ddof:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.var(axis=None, dtype=None, out=None, ddof=0, keepdims=False)

Returns the variance of the array elements, along given axis.

Refer to `numpy.var` for full documentation.

See Also
--------
numpy.var : equivalent function
*)

val view : ?dtype:[`Ndarray_sub_class of Py.Object.t | `Dtype of Np.Dtype.t] -> ?type_:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.view([dtype][, type])

New view of array with the same data.

.. note::
    Passing None for ``dtype`` is different from omitting the parameter,
    since the former invokes ``dtype(None)`` which is an alias for
    ``dtype('float_')``.

Parameters
----------
dtype : data-type or ndarray sub-class, optional
    Data-type descriptor of the returned view, e.g., float32 or int16.
    Omitting it results in the view having the same data-type as `a`.
    This argument can also be specified as an ndarray sub-class, which
    then specifies the type of the returned object (this is equivalent to
    setting the ``type`` parameter).
type : Python type, optional
    Type of the returned view, e.g., ndarray or matrix.  Again, omission
    of the parameter results in type preservation.

Notes
-----
``a.view()`` is used two different ways:

``a.view(some_dtype)`` or ``a.view(dtype=some_dtype)`` constructs a view
of the array's memory with a different data-type.  This can cause a
reinterpretation of the bytes of memory.

``a.view(ndarray_subclass)`` or ``a.view(type=ndarray_subclass)`` just
returns an instance of `ndarray_subclass` that looks at the same array
(same shape, dtype, etc.)  This does not cause a reinterpretation of the
memory.

For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of
bytes per entry than the previous dtype (for example, converting a
regular array to a structured array), then the behavior of the view
cannot be predicted just from the superficial appearance of ``a`` (shown
by ``print(a)``). It also depends on exactly how ``a`` is stored in
memory. Therefore if ``a`` is C-ordered versus fortran-ordered, versus
defined as a slice or transpose, etc., the view may give different
results.


Examples
--------
>>> x = np.array([(1, 2)], dtype=[('a', np.int8), ('b', np.int8)])

Viewing array data using a different type and dtype:

>>> y = x.view(dtype=np.int16, type=np.matrix)
>>> y
matrix([[513]], dtype=int16)
>>> print(type(y))
<class 'numpy.matrix'>

Creating a view on a structured array so it can be used in calculations

>>> x = np.array([(1, 2),(3,4)], dtype=[('a', np.int8), ('b', np.int8)])
>>> xv = x.view(dtype=np.int8).reshape(-1,2)
>>> xv
array([[1, 2],
       [3, 4]], dtype=int8)
>>> xv.mean(0)
array([2.,  3.])

Making changes to the view changes the underlying array

>>> xv[0,1] = 20
>>> x
array([(1, 20), (3,  4)], dtype=[('a', 'i1'), ('b', 'i1')])

Using a view to convert an array to a recarray:

>>> z = x.view(np.recarray)
>>> z.a
array([1, 3], dtype=int8)

Views share data:

>>> x[0] = (9, 10)
>>> z[0]
(9, 10)

Views that change the dtype size (bytes per entry) should normally be
avoided on arrays defined by slices, transposes, fortran-ordering, etc.:

>>> x = np.array([[1,2,3],[4,5,6]], dtype=np.int16)
>>> y = x[:, 0:2]
>>> y
array([[1, 2],
       [4, 5]], dtype=int16)
>>> y.view(dtype=[('width', np.int16), ('length', np.int16)])
Traceback (most recent call last):
    ...
ValueError: To change to a dtype of a different size, the array must be C-contiguous
>>> z = y.copy()
>>> z.view(dtype=[('width', np.int16), ('length', np.int16)])
array([[(1, 2)],
       [(4, 5)]], dtype=[('width', '<i2'), ('length', '<i2')])
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module MatlabObject : sig
type tag = [`MatlabObject]
type t = [`ArrayLike | `MatlabObject | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?classname:Py.Object.t -> input_array:Py.Object.t -> unit -> t
(**
ndarray Subclass to contain matlab object 
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)

val __setitem__ : key:Py.Object.t -> value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Set self[key] to value.
*)

val all : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.all(axis=None, out=None, keepdims=False)

Returns True if all elements evaluate to True.

Refer to `numpy.all` for full documentation.

See Also
--------
numpy.all : equivalent function
*)

val any : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.any(axis=None, out=None, keepdims=False)

Returns True if any of the elements of `a` evaluate to True.

Refer to `numpy.any` for full documentation.

See Also
--------
numpy.any : equivalent function
*)

val argmax : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argmax(axis=None, out=None)

Return indices of the maximum values along the given axis.

Refer to `numpy.argmax` for full documentation.

See Also
--------
numpy.argmax : equivalent function
*)

val argmin : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argmin(axis=None, out=None)

Return indices of the minimum values along the given axis of `a`.

Refer to `numpy.argmin` for detailed documentation.

See Also
--------
numpy.argmin : equivalent function
*)

val argpartition : ?axis:Py.Object.t -> ?kind:Py.Object.t -> ?order:Py.Object.t -> kth:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argpartition(kth, axis=-1, kind='introselect', order=None)

Returns the indices that would partition this array.

Refer to `numpy.argpartition` for full documentation.

.. versionadded:: 1.8.0

See Also
--------
numpy.argpartition : equivalent function
*)

val argsort : ?axis:Py.Object.t -> ?kind:Py.Object.t -> ?order:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argsort(axis=-1, kind=None, order=None)

Returns the indices that would sort this array.

Refer to `numpy.argsort` for full documentation.

See Also
--------
numpy.argsort : equivalent function
*)

val astype : ?order:[`C | `F | `A | `K] -> ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?subok:Py.Object.t -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)

Copy of the array, cast to a specified type.

Parameters
----------
dtype : str or dtype
    Typecode or data-type to which the array is cast.
order : {'C', 'F', 'A', 'K'}, optional
    Controls the memory layout order of the result.
    'C' means C order, 'F' means Fortran order, 'A'
    means 'F' order if all the arrays are Fortran contiguous,
    'C' order otherwise, and 'K' means as close to the
    order the array elements appear in memory as possible.
    Default is 'K'.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur. Defaults to 'unsafe'
    for backwards compatibility.

      * 'no' means the data types should not be cast at all.
      * 'equiv' means only byte-order changes are allowed.
      * 'safe' means only casts which can preserve values are allowed.
      * 'same_kind' means only safe casts or casts within a kind,
        like float64 to float32, are allowed.
      * 'unsafe' means any data conversions may be done.
subok : bool, optional
    If True, then sub-classes will be passed-through (default), otherwise
    the returned array will be forced to be a base-class array.
copy : bool, optional
    By default, astype always returns a newly allocated array. If this
    is set to false, and the `dtype`, `order`, and `subok`
    requirements are satisfied, the input array is returned instead
    of a copy.

Returns
-------
arr_t : ndarray
    Unless `copy` is False and the other conditions for returning the input
    array are satisfied (see description for `copy` input parameter), `arr_t`
    is a new array of the same shape as the input array, with dtype, order
    given by `dtype`, `order`.

Notes
-----
.. versionchanged:: 1.17.0
   Casting between a simple data type and a structured one is possible only
   for 'unsafe' casting.  Casting to multiple fields is allowed, but
   casting from multiple fields is not.

.. versionchanged:: 1.9.0
   Casting from numeric to string types in 'safe' casting mode requires
   that the string dtype length is long enough to store the max
   integer/float value converted.

Raises
------
ComplexWarning
    When casting from complex to float or int. To avoid this,
    one should use ``a.real.astype(t)``.

Examples
--------
>>> x = np.array([1, 2, 2.5])
>>> x
array([1. ,  2. ,  2.5])

>>> x.astype(int)
array([1, 2, 2])
*)

val byteswap : ?inplace:bool -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.byteswap(inplace=False)

Swap the bytes of the array elements

Toggle between low-endian and big-endian data representation by
returning a byteswapped array, optionally swapped in-place.
Arrays of byte-strings are not swapped. The real and imaginary
parts of a complex number are swapped individually.

Parameters
----------
inplace : bool, optional
    If ``True``, swap bytes in-place, default is ``False``.

Returns
-------
out : ndarray
    The byteswapped array. If `inplace` is ``True``, this is
    a view to self.

Examples
--------
>>> A = np.array([1, 256, 8755], dtype=np.int16)
>>> list(map(hex, A))
['0x1', '0x100', '0x2233']
>>> A.byteswap(inplace=True)
array([  256,     1, 13090], dtype=int16)
>>> list(map(hex, A))
['0x100', '0x1', '0x3322']

Arrays of byte-strings are not swapped

>>> A = np.array([b'ceg', b'fac'])
>>> A.byteswap()
array([b'ceg', b'fac'], dtype='|S3')

``A.newbyteorder().byteswap()`` produces an array with the same values
  but different representation in memory

>>> A = np.array([1, 2, 3])
>>> A.view(np.uint8)
array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
       0, 0], dtype=uint8)
>>> A.newbyteorder().byteswap(inplace=True)
array([1, 2, 3])
>>> A.view(np.uint8)
array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
       0, 3], dtype=uint8)
*)

val choose : ?out:Py.Object.t -> ?mode:Py.Object.t -> choices:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.choose(choices, out=None, mode='raise')

Use an index array to construct a new array from a set of choices.

Refer to `numpy.choose` for full documentation.

See Also
--------
numpy.choose : equivalent function
*)

val clip : ?min:Py.Object.t -> ?max:Py.Object.t -> ?out:Py.Object.t -> ?kwargs:(string * Py.Object.t) list -> [> tag] Obj.t -> Py.Object.t
(**
a.clip(min=None, max=None, out=None, **kwargs)

Return an array whose values are limited to ``[min, max]``.
One of max or min must be given.

Refer to `numpy.clip` for full documentation.

See Also
--------
numpy.clip : equivalent function
*)

val compress : ?axis:Py.Object.t -> ?out:Py.Object.t -> condition:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.compress(condition, axis=None, out=None)

Return selected slices of this array along given axis.

Refer to `numpy.compress` for full documentation.

See Also
--------
numpy.compress : equivalent function
*)

val conj : [> tag] Obj.t -> Py.Object.t
(**
a.conj()

Complex-conjugate all elements.

Refer to `numpy.conjugate` for full documentation.

See Also
--------
numpy.conjugate : equivalent function
*)

val conjugate : [> tag] Obj.t -> Py.Object.t
(**
a.conjugate()

Return the complex conjugate, element-wise.

Refer to `numpy.conjugate` for full documentation.

See Also
--------
numpy.conjugate : equivalent function
*)

val copy : ?order:[`C | `F | `A | `K] -> [> tag] Obj.t -> Py.Object.t
(**
a.copy(order='C')

Return a copy of the array.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    Controls the memory layout of the copy. 'C' means C-order,
    'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
    'C' otherwise. 'K' means match the layout of `a` as closely
    as possible. (Note that this function and :func:`numpy.copy` are very
    similar, but have different default values for their order=
    arguments.)

See also
--------
numpy.copy
numpy.copyto

Examples
--------
>>> x = np.array([[1,2,3],[4,5,6]], order='F')

>>> y = x.copy()

>>> x.fill(0)

>>> x
array([[0, 0, 0],
       [0, 0, 0]])

>>> y
array([[1, 2, 3],
       [4, 5, 6]])

>>> y.flags['C_CONTIGUOUS']
True
*)

val cumprod : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.cumprod(axis=None, dtype=None, out=None)

Return the cumulative product of the elements along the given axis.

Refer to `numpy.cumprod` for full documentation.

See Also
--------
numpy.cumprod : equivalent function
*)

val cumsum : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.cumsum(axis=None, dtype=None, out=None)

Return the cumulative sum of the elements along the given axis.

Refer to `numpy.cumsum` for full documentation.

See Also
--------
numpy.cumsum : equivalent function
*)

val diagonal : ?offset:Py.Object.t -> ?axis1:Py.Object.t -> ?axis2:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.diagonal(offset=0, axis1=0, axis2=1)

Return specified diagonals. In NumPy 1.9 the returned array is a
read-only view instead of a copy as in previous NumPy versions.  In
a future version the read-only restriction will be removed.

Refer to :func:`numpy.diagonal` for full documentation.

See Also
--------
numpy.diagonal : equivalent function
*)

val dot : ?out:Py.Object.t -> b:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.dot(b, out=None)

Dot product of two arrays.

Refer to `numpy.dot` for full documentation.

See Also
--------
numpy.dot : equivalent function

Examples
--------
>>> a = np.eye(2)
>>> b = np.ones((2, 2)) * 2
>>> a.dot(b)
array([[2.,  2.],
       [2.,  2.]])

This array method can be conveniently chained:

>>> a.dot(b).dot(b)
array([[8.,  8.],
       [8.,  8.]])
*)

val dump : file:[`S of string | `Path of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.dump(file)

Dump a pickle of the array to the specified file.
The array can be read back with pickle.load or numpy.load.

Parameters
----------
file : str or Path
    A string naming the dump file.

    .. versionchanged:: 1.17.0
        `pathlib.Path` objects are now accepted.
*)

val dumps : [> tag] Obj.t -> Py.Object.t
(**
a.dumps()

Returns the pickle of the array as a string.
pickle.loads or numpy.loads will convert the string back to an array.

Parameters
----------
None
*)

val fill : value:[`F of float | `I of int | `Bool of bool | `S of string] -> [> tag] Obj.t -> Py.Object.t
(**
a.fill(value)

Fill the array with a scalar value.

Parameters
----------
value : scalar
    All elements of `a` will be assigned this value.

Examples
--------
>>> a = np.array([1, 2])
>>> a.fill(0)
>>> a
array([0, 0])
>>> a = np.empty(2)
>>> a.fill(1)
>>> a
array([1.,  1.])
*)

val flatten : ?order:[`C | `F | `A | `K] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.flatten(order='C')

Return a copy of the array collapsed into one dimension.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    'C' means to flatten in row-major (C-style) order.
    'F' means to flatten in column-major (Fortran-
    style) order. 'A' means to flatten in column-major
    order if `a` is Fortran *contiguous* in memory,
    row-major order otherwise. 'K' means to flatten
    `a` in the order the elements occur in memory.
    The default is 'C'.

Returns
-------
y : ndarray
    A copy of the input array, flattened to one dimension.

See Also
--------
ravel : Return a flattened array.
flat : A 1-D flat iterator over the array.

Examples
--------
>>> a = np.array([[1,2], [3,4]])
>>> a.flatten()
array([1, 2, 3, 4])
>>> a.flatten('F')
array([1, 3, 2, 4])
*)

val getfield : ?offset:int -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.getfield(dtype, offset=0)

Returns a field of the given array as a certain type.

A field is a view of the array data with a given data-type. The values in
the view are determined by the given type and the offset into the current
array in bytes. The offset needs to be such that the view dtype fits in the
array dtype; for example an array of dtype complex128 has 16-byte elements.
If taking a view with a 32-bit integer (4 bytes), the offset needs to be
between 0 and 12 bytes.

Parameters
----------
dtype : str or dtype
    The data type of the view. The dtype size of the view can not be larger
    than that of the array itself.
offset : int
    Number of bytes to skip before beginning the element view.

Examples
--------
>>> x = np.diag([1.+1.j]*2)
>>> x[1, 1] = 2 + 4.j
>>> x
array([[1.+1.j,  0.+0.j],
       [0.+0.j,  2.+4.j]])
>>> x.getfield(np.float64)
array([[1.,  0.],
       [0.,  2.]])

By choosing an offset of 8 bytes we can select the complex part of the
array for our view:

>>> x.getfield(np.float64, offset=8)
array([[1.,  0.],
       [0.,  4.]])
*)

val item : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
a.item( *args)

Copy an element of an array to a standard Python scalar and return it.

Parameters
----------
\*args : Arguments (variable number and type)

    * none: in this case, the method only works for arrays
      with one element (`a.size == 1`), which element is
      copied into a standard Python scalar object and returned.

    * int_type: this argument is interpreted as a flat index into
      the array, specifying which element to copy and return.

    * tuple of int_types: functions as does a single int_type argument,
      except that the argument is interpreted as an nd-index into the
      array.

Returns
-------
z : Standard Python scalar object
    A copy of the specified element of the array as a suitable
    Python scalar

Notes
-----
When the data type of `a` is longdouble or clongdouble, item() returns
a scalar array object because there is no available Python scalar that
would not lose information. Void arrays return a buffer object for item(),
unless fields are defined, in which case a tuple is returned.

`item` is very similar to a[args], except, instead of an array scalar,
a standard Python scalar is returned. This can be useful for speeding up
access to elements of the array and doing arithmetic on elements of the
array using Python's optimized math.

Examples
--------
>>> np.random.seed(123)
>>> x = np.random.randint(9, size=(3, 3))
>>> x
array([[2, 2, 6],
       [1, 3, 6],
       [1, 0, 1]])
>>> x.item(3)
1
>>> x.item(7)
0
>>> x.item((0, 1))
2
>>> x.item((2, 2))
1
*)

val itemset : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
a.itemset( *args)

Insert scalar into an array (scalar is cast to array's dtype, if possible)

There must be at least 1 argument, and define the last argument
as *item*.  Then, ``a.itemset( *args)`` is equivalent to but faster
than ``a[args] = item``.  The item should be a scalar value and `args`
must select a single item in the array `a`.

Parameters
----------
\*args : Arguments
    If one argument: a scalar, only used in case `a` is of size 1.
    If two arguments: the last argument is the value to be set
    and must be a scalar, the first argument specifies a single array
    element location. It is either an int or a tuple.

Notes
-----
Compared to indexing syntax, `itemset` provides some speed increase
for placing a scalar into a particular location in an `ndarray`,
if you must do this.  However, generally this is discouraged:
among other problems, it complicates the appearance of the code.
Also, when using `itemset` (and `item`) inside a loop, be sure
to assign the methods to a local variable to avoid the attribute
look-up at each loop iteration.

Examples
--------
>>> np.random.seed(123)
>>> x = np.random.randint(9, size=(3, 3))
>>> x
array([[2, 2, 6],
       [1, 3, 6],
       [1, 0, 1]])
>>> x.itemset(4, 0)
>>> x.itemset((2, 2), 9)
>>> x
array([[2, 2, 6],
       [1, 0, 6],
       [1, 0, 9]])
*)

val max : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> ?initial:Py.Object.t -> ?where:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

Return the maximum along a given axis.

Refer to `numpy.amax` for full documentation.

See Also
--------
numpy.amax : equivalent function
*)

val mean : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.mean(axis=None, dtype=None, out=None, keepdims=False)

Returns the average of the array elements along given axis.

Refer to `numpy.mean` for full documentation.

See Also
--------
numpy.mean : equivalent function
*)

val min : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> ?initial:Py.Object.t -> ?where:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

Return the minimum along a given axis.

Refer to `numpy.amin` for full documentation.

See Also
--------
numpy.amin : equivalent function
*)

val newbyteorder : ?new_order:string -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
arr.newbyteorder(new_order='S')

Return the array with the same data viewed with a different byte order.

Equivalent to::

    arr.view(arr.dtype.newbytorder(new_order))

Changes are also made in all fields and sub-arrays of the array data
type.



Parameters
----------
new_order : string, optional
    Byte order to force; a value from the byte order specifications
    below. `new_order` codes can be any of:

    * 'S' - swap dtype from current to opposite endian
    * {'<', 'L'} - little endian
    * {'>', 'B'} - big endian
    * {'=', 'N'} - native order
    * {'|', 'I'} - ignore (no change to byte order)

    The default value ('S') results in swapping the current
    byte order. The code does a case-insensitive check on the first
    letter of `new_order` for the alternatives above.  For example,
    any of 'B' or 'b' or 'biggish' are valid to specify big-endian.


Returns
-------
new_arr : array
    New array object with the dtype reflecting given change to the
    byte order.
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
(**
a.nonzero()

Return the indices of the elements that are non-zero.

Refer to `numpy.nonzero` for full documentation.

See Also
--------
numpy.nonzero : equivalent function
*)

val partition : ?axis:int -> ?kind:[`Introselect] -> ?order:[`S of string | `StringList of string list] -> kth:[`I of int | `Is of int list] -> [> tag] Obj.t -> Py.Object.t
(**
a.partition(kth, axis=-1, kind='introselect', order=None)

Rearranges the elements in the array in such a way that the value of the
element in kth position is in the position it would be in a sorted array.
All elements smaller than the kth element are moved before this element and
all equal or greater are moved behind it. The ordering of the elements in
the two partitions is undefined.

.. versionadded:: 1.8.0

Parameters
----------
kth : int or sequence of ints
    Element index to partition by. The kth element value will be in its
    final sorted position and all smaller elements will be moved before it
    and all equal or greater elements behind it.
    The order of all elements in the partitions is undefined.
    If provided with a sequence of kth it will partition all elements
    indexed by kth of them into their sorted position at once.
axis : int, optional
    Axis along which to sort. Default is -1, which means sort along the
    last axis.
kind : {'introselect'}, optional
    Selection algorithm. Default is 'introselect'.
order : str or list of str, optional
    When `a` is an array with fields defined, this argument specifies
    which fields to compare first, second, etc. A single field can
    be specified as a string, and not all fields need to be specified,
    but unspecified fields will still be used, in the order in which
    they come up in the dtype, to break ties.

See Also
--------
numpy.partition : Return a parititioned copy of an array.
argpartition : Indirect partition.
sort : Full sort.

Notes
-----
See ``np.partition`` for notes on the different algorithms.

Examples
--------
>>> a = np.array([3, 4, 2, 1])
>>> a.partition(3)
>>> a
array([2, 1, 3, 4])

>>> a.partition((1, 3))
>>> a
array([1, 2, 3, 4])
*)

val prod : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> ?initial:Py.Object.t -> ?where:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.prod(axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True)

Return the product of the array elements over the given axis

Refer to `numpy.prod` for full documentation.

See Also
--------
numpy.prod : equivalent function
*)

val ptp : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.ptp(axis=None, out=None, keepdims=False)

Peak to peak (maximum - minimum) value along a given axis.

Refer to `numpy.ptp` for full documentation.

See Also
--------
numpy.ptp : equivalent function
*)

val put : ?mode:Py.Object.t -> indices:Py.Object.t -> values:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.put(indices, values, mode='raise')

Set ``a.flat[n] = values[n]`` for all `n` in indices.

Refer to `numpy.put` for full documentation.

See Also
--------
numpy.put : equivalent function
*)

val ravel : ?order:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.ravel([order])

Return a flattened array.

Refer to `numpy.ravel` for full documentation.

See Also
--------
numpy.ravel : equivalent function

ndarray.flat : a flat iterator on the array.
*)

val repeat : ?axis:Py.Object.t -> repeats:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.repeat(repeats, axis=None)

Repeat elements of an array.

Refer to `numpy.repeat` for full documentation.

See Also
--------
numpy.repeat : equivalent function
*)

val reshape : ?order:Py.Object.t -> shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.reshape(shape, order='C')

Returns an array containing the same data with a new shape.

Refer to `numpy.reshape` for full documentation.

See Also
--------
numpy.reshape : equivalent function

Notes
-----
Unlike the free function `numpy.reshape`, this method on `ndarray` allows
the elements of the shape parameter to be passed in as separate arguments.
For example, ``a.reshape(10, 11)`` is equivalent to
``a.reshape((10, 11))``.
*)

val resize : ?refcheck:bool -> new_shape:[`T_n_ints of Py.Object.t | `TupleOfInts of int list] -> [> tag] Obj.t -> Py.Object.t
(**
a.resize(new_shape, refcheck=True)

Change shape and size of array in-place.

Parameters
----------
new_shape : tuple of ints, or `n` ints
    Shape of resized array.
refcheck : bool, optional
    If False, reference count will not be checked. Default is True.

Returns
-------
None

Raises
------
ValueError
    If `a` does not own its own data or references or views to it exist,
    and the data memory must be changed.
    PyPy only: will always raise if the data memory must be changed, since
    there is no reliable way to determine if references or views to it
    exist.

SystemError
    If the `order` keyword argument is specified. This behaviour is a
    bug in NumPy.

See Also
--------
resize : Return a new array with the specified shape.

Notes
-----
This reallocates space for the data area if necessary.

Only contiguous arrays (data elements consecutive in memory) can be
resized.

The purpose of the reference count check is to make sure you
do not use this array as a buffer for another Python object and then
reallocate the memory. However, reference counts can increase in
other ways so if you are sure that you have not shared the memory
for this array with another Python object, then you may safely set
`refcheck` to False.

Examples
--------
Shrinking an array: array is flattened (in the order that the data are
stored in memory), resized, and reshaped:

>>> a = np.array([[0, 1], [2, 3]], order='C')
>>> a.resize((2, 1))
>>> a
array([[0],
       [1]])

>>> a = np.array([[0, 1], [2, 3]], order='F')
>>> a.resize((2, 1))
>>> a
array([[0],
       [2]])

Enlarging an array: as above, but missing entries are filled with zeros:

>>> b = np.array([[0, 1], [2, 3]])
>>> b.resize(2, 3) # new_shape parameter doesn't have to be a tuple
>>> b
array([[0, 1, 2],
       [3, 0, 0]])

Referencing an array prevents resizing...

>>> c = a
>>> a.resize((1, 1))
Traceback (most recent call last):
...
ValueError: cannot resize an array that references or is referenced ...

Unless `refcheck` is False:

>>> a.resize((1, 1), refcheck=False)
>>> a
array([[0]])
>>> c
array([[0]])
*)

val round : ?decimals:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.round(decimals=0, out=None)

Return `a` with each element rounded to the given number of decimals.

Refer to `numpy.around` for full documentation.

See Also
--------
numpy.around : equivalent function
*)

val searchsorted : ?side:Py.Object.t -> ?sorter:Py.Object.t -> v:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.searchsorted(v, side='left', sorter=None)

Find indices where elements of v should be inserted in a to maintain order.

For full documentation, see `numpy.searchsorted`

See Also
--------
numpy.searchsorted : equivalent function
*)

val setfield : ?offset:int -> val_:Py.Object.t -> dtype:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.setfield(val, dtype, offset=0)

Put a value into a specified place in a field defined by a data-type.

Place `val` into `a`'s field defined by `dtype` and beginning `offset`
bytes into the field.

Parameters
----------
val : object
    Value to be placed in field.
dtype : dtype object
    Data-type of the field in which to place `val`.
offset : int, optional
    The number of bytes into the field at which to place `val`.

Returns
-------
None

See Also
--------
getfield

Examples
--------
>>> x = np.eye(3)
>>> x.getfield(np.float64)
array([[1.,  0.,  0.],
       [0.,  1.,  0.],
       [0.,  0.,  1.]])
>>> x.setfield(3, np.int32)
>>> x.getfield(np.int32)
array([[3, 3, 3],
       [3, 3, 3],
       [3, 3, 3]], dtype=int32)
>>> x
array([[1.0e+000, 1.5e-323, 1.5e-323],
       [1.5e-323, 1.0e+000, 1.5e-323],
       [1.5e-323, 1.5e-323, 1.0e+000]])
>>> x.setfield(np.eye(3), np.int32)
>>> x
array([[1.,  0.,  0.],
       [0.,  1.,  0.],
       [0.,  0.,  1.]])
*)

val setflags : ?write:bool -> ?align:bool -> ?uic:bool -> [> tag] Obj.t -> Py.Object.t
(**
a.setflags(write=None, align=None, uic=None)

Set array flags WRITEABLE, ALIGNED, (WRITEBACKIFCOPY and UPDATEIFCOPY),
respectively.

These Boolean-valued flags affect how numpy interprets the memory
area used by `a` (see Notes below). The ALIGNED flag can only
be set to True if the data is actually aligned according to the type.
The WRITEBACKIFCOPY and (deprecated) UPDATEIFCOPY flags can never be set
to True. The flag WRITEABLE can only be set to True if the array owns its
own memory, or the ultimate owner of the memory exposes a writeable buffer
interface, or is a string. (The exception for string is made so that
unpickling can be done without copying memory.)

Parameters
----------
write : bool, optional
    Describes whether or not `a` can be written to.
align : bool, optional
    Describes whether or not `a` is aligned properly for its type.
uic : bool, optional
    Describes whether or not `a` is a copy of another 'base' array.

Notes
-----
Array flags provide information about how the memory area used
for the array is to be interpreted. There are 7 Boolean flags
in use, only four of which can be changed by the user:
WRITEBACKIFCOPY, UPDATEIFCOPY, WRITEABLE, and ALIGNED.

WRITEABLE (W) the data area can be written to;

ALIGNED (A) the data and strides are aligned appropriately for the hardware
(as determined by the compiler);

UPDATEIFCOPY (U) (deprecated), replaced by WRITEBACKIFCOPY;

WRITEBACKIFCOPY (X) this array is a copy of some other array (referenced
by .base). When the C-API function PyArray_ResolveWritebackIfCopy is
called, the base array will be updated with the contents of this array.

All flags can be accessed using the single (upper case) letter as well
as the full name.

Examples
--------
>>> y = np.array([[3, 1, 7],
...               [2, 0, 0],
...               [8, 5, 9]])
>>> y
array([[3, 1, 7],
       [2, 0, 0],
       [8, 5, 9]])
>>> y.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : True
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
>>> y.setflags(write=0, align=0)
>>> y.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : True
  WRITEABLE : False
  ALIGNED : False
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
>>> y.setflags(uic=1)
Traceback (most recent call last):
  File '<stdin>', line 1, in <module>
ValueError: cannot set WRITEBACKIFCOPY flag to True
*)

val sort : ?axis:int -> ?kind:[`Quicksort | `Heapsort | `Stable | `Mergesort] -> ?order:[`S of string | `StringList of string list] -> [> tag] Obj.t -> Py.Object.t
(**
a.sort(axis=-1, kind=None, order=None)

Sort an array in-place. Refer to `numpy.sort` for full documentation.

Parameters
----------
axis : int, optional
    Axis along which to sort. Default is -1, which means sort along the
    last axis.
kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
    Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
    and 'mergesort' use timsort under the covers and, in general, the
    actual implementation will vary with datatype. The 'mergesort' option
    is retained for backwards compatibility.

    .. versionchanged:: 1.15.0.
       The 'stable' option was added.

order : str or list of str, optional
    When `a` is an array with fields defined, this argument specifies
    which fields to compare first, second, etc.  A single field can
    be specified as a string, and not all fields need be specified,
    but unspecified fields will still be used, in the order in which
    they come up in the dtype, to break ties.

See Also
--------
numpy.sort : Return a sorted copy of an array.
numpy.argsort : Indirect sort.
numpy.lexsort : Indirect stable sort on multiple keys.
numpy.searchsorted : Find elements in sorted array.
numpy.partition: Partial sort.

Notes
-----
See `numpy.sort` for notes on the different sorting algorithms.

Examples
--------
>>> a = np.array([[1,4], [3,1]])
>>> a.sort(axis=1)
>>> a
array([[1, 4],
       [1, 3]])
>>> a.sort(axis=0)
>>> a
array([[1, 3],
       [1, 4]])

Use the `order` keyword to specify a field to use when sorting a
structured array:

>>> a = np.array([('a', 2), ('c', 1)], dtype=[('x', 'S1'), ('y', int)])
>>> a.sort(order='y')
>>> a
array([(b'c', 1), (b'a', 2)],
      dtype=[('x', 'S1'), ('y', '<i8')])
*)

val squeeze : ?axis:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.squeeze(axis=None)

Remove single-dimensional entries from the shape of `a`.

Refer to `numpy.squeeze` for full documentation.

See Also
--------
numpy.squeeze : equivalent function
*)

val std : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?ddof:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.std(axis=None, dtype=None, out=None, ddof=0, keepdims=False)

Returns the standard deviation of the array elements along given axis.

Refer to `numpy.std` for full documentation.

See Also
--------
numpy.std : equivalent function
*)

val sum : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> ?initial:Py.Object.t -> ?where:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.sum(axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True)

Return the sum of the array elements over the given axis.

Refer to `numpy.sum` for full documentation.

See Also
--------
numpy.sum : equivalent function
*)

val swapaxes : axis1:Py.Object.t -> axis2:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.swapaxes(axis1, axis2)

Return a view of the array with `axis1` and `axis2` interchanged.

Refer to `numpy.swapaxes` for full documentation.

See Also
--------
numpy.swapaxes : equivalent function
*)

val take : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?mode:Py.Object.t -> indices:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.take(indices, axis=None, out=None, mode='raise')

Return an array formed from the elements of `a` at the given indices.

Refer to `numpy.take` for full documentation.

See Also
--------
numpy.take : equivalent function
*)

val tobytes : ?order:[`C | `F | `None] -> [> tag] Obj.t -> Py.Object.t
(**
a.tobytes(order='C')

Construct Python bytes containing the raw data bytes in the array.

Constructs Python bytes showing a copy of the raw contents of
data memory. The bytes object can be produced in either 'C' or 'Fortran',
or 'Any' order (the default is 'C'-order). 'Any' order means C-order
unless the F_CONTIGUOUS flag in the array is set, in which case it
means 'Fortran' order.

.. versionadded:: 1.9.0

Parameters
----------
order : {'C', 'F', None}, optional
    Order of the data for multidimensional arrays:
    C, Fortran, or the same as for the original array.

Returns
-------
s : bytes
    Python bytes exhibiting a copy of `a`'s raw data.

Examples
--------
>>> x = np.array([[0, 1], [2, 3]], dtype='<u2')
>>> x.tobytes()
b'\x00\x00\x01\x00\x02\x00\x03\x00'
>>> x.tobytes('C') == x.tobytes()
True
>>> x.tobytes('F')
b'\x00\x00\x02\x00\x01\x00\x03\x00'
*)

val tofile : ?sep:string -> ?format:string -> fid:[`S of string | `PyObject of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.tofile(fid, sep='', format='%s')

Write array to a file as text or binary (default).

Data is always written in 'C' order, independent of the order of `a`.
The data produced by this method can be recovered using the function
fromfile().

Parameters
----------
fid : file or str or Path
    An open file object, or a string containing a filename.

    .. versionchanged:: 1.17.0
        `pathlib.Path` objects are now accepted.

sep : str
    Separator between array items for text output.
    If '' (empty), a binary file is written, equivalent to
    ``file.write(a.tobytes())``.
format : str
    Format string for text file output.
    Each entry in the array is formatted to text by first converting
    it to the closest Python type, and then using 'format' % item.

Notes
-----
This is a convenience function for quick storage of array data.
Information on endianness and precision is lost, so this method is not a
good choice for files intended to archive data or transport data between
machines with different endianness. Some of these problems can be overcome
by outputting the data as text files, at the expense of speed and file
size.

When fid is a file object, array contents are directly written to the
file, bypassing the file object's ``write`` method. As a result, tofile
cannot be used with files objects supporting compression (e.g., GzipFile)
or file-like objects that do not support ``fileno()`` (e.g., BytesIO).
*)

val tolist : [> tag] Obj.t -> Py.Object.t
(**
a.tolist()

Return the array as an ``a.ndim``-levels deep nested list of Python scalars.

Return a copy of the array data as a (nested) Python list.
Data items are converted to the nearest compatible builtin Python type, via
the `~numpy.ndarray.item` function.

If ``a.ndim`` is 0, then since the depth of the nested list is 0, it will
not be a list at all, but a simple Python scalar.

Parameters
----------
none

Returns
-------
y : object, or list of object, or list of list of object, or ...
    The possibly nested list of array elements.

Notes
-----
The array may be recreated via ``a = np.array(a.tolist())``, although this
may sometimes lose precision.

Examples
--------
For a 1D array, ``a.tolist()`` is almost the same as ``list(a)``,
except that ``tolist`` changes numpy scalars to Python scalars:

>>> a = np.uint32([1, 2])
>>> a_list = list(a)
>>> a_list
[1, 2]
>>> type(a_list[0])
<class 'numpy.uint32'>
>>> a_tolist = a.tolist()
>>> a_tolist
[1, 2]
>>> type(a_tolist[0])
<class 'int'>

Additionally, for a 2D array, ``tolist`` applies recursively:

>>> a = np.array([[1, 2], [3, 4]])
>>> list(a)
[array([1, 2]), array([3, 4])]
>>> a.tolist()
[[1, 2], [3, 4]]

The base case for this recursion is a 0D array:

>>> a = np.array(1)
>>> list(a)
Traceback (most recent call last):
  ...
TypeError: iteration over a 0-d array
>>> a.tolist()
1
*)

val tostring : ?order:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.tostring(order='C')

A compatibility alias for `tobytes`, with exactly the same behavior.

Despite its name, it returns `bytes` not `str`\ s.

.. deprecated:: 1.19.0
*)

val trace : ?offset:Py.Object.t -> ?axis1:Py.Object.t -> ?axis2:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)

Return the sum along diagonals of the array.

Refer to `numpy.trace` for full documentation.

See Also
--------
numpy.trace : equivalent function
*)

val transpose : Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.transpose( *axes)

Returns a view of the array with axes transposed.

For a 1-D array this has no effect, as a transposed vector is simply the
same vector. To convert a 1-D array into a 2D column vector, an additional
dimension must be added. `np.atleast2d(a).T` achieves this, as does
`a[:, np.newaxis]`.
For a 2-D array, this is a standard matrix transpose.
For an n-D array, if axes are given, their order indicates how the
axes are permuted (see Examples). If axes are not provided and
``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

Parameters
----------
axes : None, tuple of ints, or `n` ints

 * None or no argument: reverses the order of the axes.

 * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
   `i`-th axis becomes `a.transpose()`'s `j`-th axis.

 * `n` ints: same as an n-tuple of the same ints (this form is
   intended simply as a 'convenience' alternative to the tuple form)

Returns
-------
out : ndarray
    View of `a`, with axes suitably permuted.

See Also
--------
ndarray.T : Array property returning the array transposed.
ndarray.reshape : Give a new shape to an array without changing its data.

Examples
--------
>>> a = np.array([[1, 2], [3, 4]])
>>> a
array([[1, 2],
       [3, 4]])
>>> a.transpose()
array([[1, 3],
       [2, 4]])
>>> a.transpose((1, 0))
array([[1, 3],
       [2, 4]])
>>> a.transpose(1, 0)
array([[1, 3],
       [2, 4]])
*)

val var : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?ddof:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.var(axis=None, dtype=None, out=None, ddof=0, keepdims=False)

Returns the variance of the array elements, along given axis.

Refer to `numpy.var` for full documentation.

See Also
--------
numpy.var : equivalent function
*)

val view : ?dtype:[`Ndarray_sub_class of Py.Object.t | `Dtype of Np.Dtype.t] -> ?type_:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.view([dtype][, type])

New view of array with the same data.

.. note::
    Passing None for ``dtype`` is different from omitting the parameter,
    since the former invokes ``dtype(None)`` which is an alias for
    ``dtype('float_')``.

Parameters
----------
dtype : data-type or ndarray sub-class, optional
    Data-type descriptor of the returned view, e.g., float32 or int16.
    Omitting it results in the view having the same data-type as `a`.
    This argument can also be specified as an ndarray sub-class, which
    then specifies the type of the returned object (this is equivalent to
    setting the ``type`` parameter).
type : Python type, optional
    Type of the returned view, e.g., ndarray or matrix.  Again, omission
    of the parameter results in type preservation.

Notes
-----
``a.view()`` is used two different ways:

``a.view(some_dtype)`` or ``a.view(dtype=some_dtype)`` constructs a view
of the array's memory with a different data-type.  This can cause a
reinterpretation of the bytes of memory.

``a.view(ndarray_subclass)`` or ``a.view(type=ndarray_subclass)`` just
returns an instance of `ndarray_subclass` that looks at the same array
(same shape, dtype, etc.)  This does not cause a reinterpretation of the
memory.

For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of
bytes per entry than the previous dtype (for example, converting a
regular array to a structured array), then the behavior of the view
cannot be predicted just from the superficial appearance of ``a`` (shown
by ``print(a)``). It also depends on exactly how ``a`` is stored in
memory. Therefore if ``a`` is C-ordered versus fortran-ordered, versus
defined as a slice or transpose, etc., the view may give different
results.


Examples
--------
>>> x = np.array([(1, 2)], dtype=[('a', np.int8), ('b', np.int8)])

Viewing array data using a different type and dtype:

>>> y = x.view(dtype=np.int16, type=np.matrix)
>>> y
matrix([[513]], dtype=int16)
>>> print(type(y))
<class 'numpy.matrix'>

Creating a view on a structured array so it can be used in calculations

>>> x = np.array([(1, 2),(3,4)], dtype=[('a', np.int8), ('b', np.int8)])
>>> xv = x.view(dtype=np.int8).reshape(-1,2)
>>> xv
array([[1, 2],
       [3, 4]], dtype=int8)
>>> xv.mean(0)
array([2.,  3.])

Making changes to the view changes the underlying array

>>> xv[0,1] = 20
>>> x
array([(1, 20), (3,  4)], dtype=[('a', 'i1'), ('b', 'i1')])

Using a view to convert an array to a recarray:

>>> z = x.view(np.recarray)
>>> z.a
array([1, 3], dtype=int8)

Views share data:

>>> x[0] = (9, 10)
>>> z[0]
(9, 10)

Views that change the dtype size (bytes per entry) should normally be
avoided on arrays defined by slices, transposes, fortran-ordering, etc.:

>>> x = np.array([[1,2,3],[4,5,6]], dtype=np.int16)
>>> y = x[:, 0:2]
>>> y
array([[1, 2],
       [4, 5]], dtype=int16)
>>> y.view(dtype=[('width', np.int16), ('length', np.int16)])
Traceback (most recent call last):
    ...
ValueError: To change to a dtype of a different size, the array must be C-contiguous
>>> z = y.copy()
>>> z.view(dtype=[('width', np.int16), ('length', np.int16)])
array([[(1, 2)],
       [(4, 5)]], dtype=[('width', '<i2'), ('length', '<i2')])
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module VarReader5 : sig
type tag = [`VarReader5]
type t = [`Object | `VarReader5] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module VarWriter5 : sig
type tag = [`VarWriter5]
type t = [`Object | `VarWriter5] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t -> t
(**
Generic matlab matrix writing class 
*)

val update_matrix_tag : start_pos:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val write : arr:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Write `arr` to stream at top and sub levels

Parameters
----------
arr : array_like
    array-like object to create writer for
*)

val write_bytes : arr:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val write_cells : arr:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val write_char : ?codec:Py.Object.t -> arr:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Write string array `arr` with given `codec`
        
*)

val write_element : ?mdtype:Py.Object.t -> arr:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
write tag and data 
*)

val write_empty_struct : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val write_header : ?is_complex:Py.Object.t -> ?is_logical:Py.Object.t -> ?nzmax:Py.Object.t -> shape:Py.Object.t -> mclass:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Write header for given data options
shape : sequence
   array shape
mclass      - mat5 matrix class
is_complex  - True if matrix is complex
is_logical  - True if matrix is logical
nzmax        - max non zero elements for sparse arrays

We get the name and the global flag from the object, and reset
them to defaults after we've used them
*)

val write_numeric : arr:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val write_object : arr:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Same as writing structs, except different mx class, and extra
classname element after header
*)

val write_regular_element : arr:Py.Object.t -> mdtype:Py.Object.t -> byte_count:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val write_smalldata_element : arr:Py.Object.t -> mdtype:Py.Object.t -> byte_count:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val write_sparse : arr:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Sparse matrices are 2D
        
*)

val write_string : s:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val write_struct : arr:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val write_top : arr:[>`Ndarray] Np.Obj.t -> name:string -> is_global:bool -> [> tag] Obj.t -> Py.Object.t
(**
Write variable at top level of mat file

Parameters
----------
arr : array_like
    array-like object to create writer for
name : str, optional
    name as it will appear in matlab workspace
    default is empty string
is_global : {False, True}, optional
    whether variable will be global on load into matlab
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module ZlibInputStream : sig
type tag = [`ZlibInputStream]
type t = [`Object | `ZlibInputStream] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Mat_struct : sig
type tag = [`Mat_struct]
type t = [`Mat_struct | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : unit -> t
(**
Placeholder for holding read data from structs

We use instances of this class when the user passes False as a value to the
``struct_as_record`` parameter of the :func:`scipy.io.matlab.loadmat`
function.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val arr_dtype_number : arr:Py.Object.t -> num:Py.Object.t -> unit -> Py.Object.t
(**
Return dtype for given number of items per element
*)

val arr_to_chars : Py.Object.t -> Py.Object.t
(**
Convert string array to char array 
*)

val asbytes : Py.Object.t -> Py.Object.t
(**
None
*)

val asstr : Py.Object.t -> Py.Object.t
(**
None
*)

val docfiller : Py.Object.t -> Py.Object.t
(**
None
*)

val matdims : ?oned_as:[`Column | `Row] -> arr:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Determine equivalent MATLAB dimensions for given array

Parameters
----------
arr : ndarray
    Input array
oned_as : {'column', 'row'}, optional
    Whether 1-D arrays are returned as MATLAB row or column matrices.
    Default is 'column'.

Returns
-------
dims : tuple
    Shape tuple, in the form MATLAB expects it.

Notes
-----
We had to decide what shape a 1 dimensional array would be by
default. ``np.atleast_2d`` thinks it is a row vector. The
default for a vector in MATLAB (e.g., ``>> 1:12``) is a row vector.

Versions of scipy up to and including 0.11 resulted (accidentally)
in 1-D arrays being read as column vectors. For the moment, we
maintain the same tradition here.

Examples
--------
>>> matdims(np.array(1)) # NumPy scalar
(1, 1)
>>> matdims(np.array([1])) # 1-D array, 1 element
(1, 1)
>>> matdims(np.array([1,2])) # 1-D array, 2 elements
(2, 1)
>>> matdims(np.array([[2],[3]])) # 2-D array, column vector
(2, 1)
>>> matdims(np.array([[2,3]])) # 2-D array, row vector
(1, 2)
>>> matdims(np.array([[[2,3]]])) # 3-D array, rowish vector
(1, 1, 2)
>>> matdims(np.array([])) # empty 1-D array
(0, 0)
>>> matdims(np.array([[]])) # empty 2-D array
(0, 0)
>>> matdims(np.array([[[]]])) # empty 3-D array
(0, 0, 0)

Optional argument flips 1-D shape behavior.

>>> matdims(np.array([1,2]), 'row') # 1-D array, 2 elements
(1, 2)

The argument has to make sense though

>>> matdims(np.array([1,2]), 'bizarre')
Traceback (most recent call last):
   ...
ValueError: 1-D option 'bizarre' is strange
*)

val read_dtype : mat_stream:Py.Object.t -> a_dtype:Np.Dtype.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Generic get of byte stream data of known type

Parameters
----------
mat_stream : file_like object
    MATLAB (tm) mat file stream
a_dtype : dtype
    dtype of array to read. `a_dtype` is assumed to be correct
    endianness.

Returns
-------
arr : ndarray
    Array of dtype `a_dtype` read from stream.
*)

val to_writeable : Py.Object.t -> Py.Object.t option
(**
Convert input object ``source`` to something we can write

Parameters
----------
source : object

Returns
-------
arr : None or ndarray or EmptyStructMarker
    If `source` cannot be converted to something we can write to a matfile,
    return None.  If `source` is equivalent to an empty dictionary, return
    ``EmptyStructMarker``.  Otherwise return `source` converted to an
    ndarray with contents for writing to matfile.
*)

val varmats_from_mat : Py.Object.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Pull variables out of mat 5 file as a sequence of mat file objects

This can be useful with a difficult mat file, containing unreadable
variables. This routine pulls the variables out in raw form and puts them,
unread, back into a file stream for saving or reading. Another use is the
pathological case where there is more than one variable of the same name in
the file; this routine returns the duplicates, whereas the standard reader
will overwrite duplicates in the returned dictionary.

The file pointer in `file_obj` will be undefined. File pointers for the
returned file-like objects are set at 0.

Parameters
----------
file_obj : file-like
    file object containing mat file

Returns
-------
named_mats : list
    list contains tuples of (name, BytesIO) where BytesIO is a file-like
    object containing mat file contents as for a single variable. The
    BytesIO contains a string with the original header and a single var. If
    ``var_file_obj`` is an individual BytesIO instance, then save as a mat
    file with something like ``open('test.mat',
    'wb').write(var_file_obj.read())``

Examples
--------
>>> import scipy.io

BytesIO is from the ``io`` module in Python 3, and is ``cStringIO`` for
Python < 3.

>>> mat_fileobj = BytesIO()
>>> scipy.io.savemat(mat_fileobj, {'b': np.arange(10), 'a': 'a string'})
>>> varmats = varmats_from_mat(mat_fileobj)
>>> sorted([name for name, str_obj in varmats])
['a', 'b']
*)


end

module Mio5_params : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module MatlabOpaque : sig
type tag = [`MatlabOpaque]
type t = [`ArrayLike | `MatlabOpaque | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t -> t
(**
Subclass to signal this is a matlab opaque matrix 
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)

val __setitem__ : key:Py.Object.t -> value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Set self[key] to value.
*)

val all : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.all(axis=None, out=None, keepdims=False)

Returns True if all elements evaluate to True.

Refer to `numpy.all` for full documentation.

See Also
--------
numpy.all : equivalent function
*)

val any : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.any(axis=None, out=None, keepdims=False)

Returns True if any of the elements of `a` evaluate to True.

Refer to `numpy.any` for full documentation.

See Also
--------
numpy.any : equivalent function
*)

val argmax : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argmax(axis=None, out=None)

Return indices of the maximum values along the given axis.

Refer to `numpy.argmax` for full documentation.

See Also
--------
numpy.argmax : equivalent function
*)

val argmin : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argmin(axis=None, out=None)

Return indices of the minimum values along the given axis of `a`.

Refer to `numpy.argmin` for detailed documentation.

See Also
--------
numpy.argmin : equivalent function
*)

val argpartition : ?axis:Py.Object.t -> ?kind:Py.Object.t -> ?order:Py.Object.t -> kth:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argpartition(kth, axis=-1, kind='introselect', order=None)

Returns the indices that would partition this array.

Refer to `numpy.argpartition` for full documentation.

.. versionadded:: 1.8.0

See Also
--------
numpy.argpartition : equivalent function
*)

val argsort : ?axis:Py.Object.t -> ?kind:Py.Object.t -> ?order:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argsort(axis=-1, kind=None, order=None)

Returns the indices that would sort this array.

Refer to `numpy.argsort` for full documentation.

See Also
--------
numpy.argsort : equivalent function
*)

val astype : ?order:[`C | `F | `A | `K] -> ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?subok:Py.Object.t -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)

Copy of the array, cast to a specified type.

Parameters
----------
dtype : str or dtype
    Typecode or data-type to which the array is cast.
order : {'C', 'F', 'A', 'K'}, optional
    Controls the memory layout order of the result.
    'C' means C order, 'F' means Fortran order, 'A'
    means 'F' order if all the arrays are Fortran contiguous,
    'C' order otherwise, and 'K' means as close to the
    order the array elements appear in memory as possible.
    Default is 'K'.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur. Defaults to 'unsafe'
    for backwards compatibility.

      * 'no' means the data types should not be cast at all.
      * 'equiv' means only byte-order changes are allowed.
      * 'safe' means only casts which can preserve values are allowed.
      * 'same_kind' means only safe casts or casts within a kind,
        like float64 to float32, are allowed.
      * 'unsafe' means any data conversions may be done.
subok : bool, optional
    If True, then sub-classes will be passed-through (default), otherwise
    the returned array will be forced to be a base-class array.
copy : bool, optional
    By default, astype always returns a newly allocated array. If this
    is set to false, and the `dtype`, `order`, and `subok`
    requirements are satisfied, the input array is returned instead
    of a copy.

Returns
-------
arr_t : ndarray
    Unless `copy` is False and the other conditions for returning the input
    array are satisfied (see description for `copy` input parameter), `arr_t`
    is a new array of the same shape as the input array, with dtype, order
    given by `dtype`, `order`.

Notes
-----
.. versionchanged:: 1.17.0
   Casting between a simple data type and a structured one is possible only
   for 'unsafe' casting.  Casting to multiple fields is allowed, but
   casting from multiple fields is not.

.. versionchanged:: 1.9.0
   Casting from numeric to string types in 'safe' casting mode requires
   that the string dtype length is long enough to store the max
   integer/float value converted.

Raises
------
ComplexWarning
    When casting from complex to float or int. To avoid this,
    one should use ``a.real.astype(t)``.

Examples
--------
>>> x = np.array([1, 2, 2.5])
>>> x
array([1. ,  2. ,  2.5])

>>> x.astype(int)
array([1, 2, 2])
*)

val byteswap : ?inplace:bool -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.byteswap(inplace=False)

Swap the bytes of the array elements

Toggle between low-endian and big-endian data representation by
returning a byteswapped array, optionally swapped in-place.
Arrays of byte-strings are not swapped. The real and imaginary
parts of a complex number are swapped individually.

Parameters
----------
inplace : bool, optional
    If ``True``, swap bytes in-place, default is ``False``.

Returns
-------
out : ndarray
    The byteswapped array. If `inplace` is ``True``, this is
    a view to self.

Examples
--------
>>> A = np.array([1, 256, 8755], dtype=np.int16)
>>> list(map(hex, A))
['0x1', '0x100', '0x2233']
>>> A.byteswap(inplace=True)
array([  256,     1, 13090], dtype=int16)
>>> list(map(hex, A))
['0x100', '0x1', '0x3322']

Arrays of byte-strings are not swapped

>>> A = np.array([b'ceg', b'fac'])
>>> A.byteswap()
array([b'ceg', b'fac'], dtype='|S3')

``A.newbyteorder().byteswap()`` produces an array with the same values
  but different representation in memory

>>> A = np.array([1, 2, 3])
>>> A.view(np.uint8)
array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
       0, 0], dtype=uint8)
>>> A.newbyteorder().byteswap(inplace=True)
array([1, 2, 3])
>>> A.view(np.uint8)
array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
       0, 3], dtype=uint8)
*)

val choose : ?out:Py.Object.t -> ?mode:Py.Object.t -> choices:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.choose(choices, out=None, mode='raise')

Use an index array to construct a new array from a set of choices.

Refer to `numpy.choose` for full documentation.

See Also
--------
numpy.choose : equivalent function
*)

val clip : ?min:Py.Object.t -> ?max:Py.Object.t -> ?out:Py.Object.t -> ?kwargs:(string * Py.Object.t) list -> [> tag] Obj.t -> Py.Object.t
(**
a.clip(min=None, max=None, out=None, **kwargs)

Return an array whose values are limited to ``[min, max]``.
One of max or min must be given.

Refer to `numpy.clip` for full documentation.

See Also
--------
numpy.clip : equivalent function
*)

val compress : ?axis:Py.Object.t -> ?out:Py.Object.t -> condition:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.compress(condition, axis=None, out=None)

Return selected slices of this array along given axis.

Refer to `numpy.compress` for full documentation.

See Also
--------
numpy.compress : equivalent function
*)

val conj : [> tag] Obj.t -> Py.Object.t
(**
a.conj()

Complex-conjugate all elements.

Refer to `numpy.conjugate` for full documentation.

See Also
--------
numpy.conjugate : equivalent function
*)

val conjugate : [> tag] Obj.t -> Py.Object.t
(**
a.conjugate()

Return the complex conjugate, element-wise.

Refer to `numpy.conjugate` for full documentation.

See Also
--------
numpy.conjugate : equivalent function
*)

val copy : ?order:[`C | `F | `A | `K] -> [> tag] Obj.t -> Py.Object.t
(**
a.copy(order='C')

Return a copy of the array.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    Controls the memory layout of the copy. 'C' means C-order,
    'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
    'C' otherwise. 'K' means match the layout of `a` as closely
    as possible. (Note that this function and :func:`numpy.copy` are very
    similar, but have different default values for their order=
    arguments.)

See also
--------
numpy.copy
numpy.copyto

Examples
--------
>>> x = np.array([[1,2,3],[4,5,6]], order='F')

>>> y = x.copy()

>>> x.fill(0)

>>> x
array([[0, 0, 0],
       [0, 0, 0]])

>>> y
array([[1, 2, 3],
       [4, 5, 6]])

>>> y.flags['C_CONTIGUOUS']
True
*)

val cumprod : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.cumprod(axis=None, dtype=None, out=None)

Return the cumulative product of the elements along the given axis.

Refer to `numpy.cumprod` for full documentation.

See Also
--------
numpy.cumprod : equivalent function
*)

val cumsum : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.cumsum(axis=None, dtype=None, out=None)

Return the cumulative sum of the elements along the given axis.

Refer to `numpy.cumsum` for full documentation.

See Also
--------
numpy.cumsum : equivalent function
*)

val diagonal : ?offset:Py.Object.t -> ?axis1:Py.Object.t -> ?axis2:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.diagonal(offset=0, axis1=0, axis2=1)

Return specified diagonals. In NumPy 1.9 the returned array is a
read-only view instead of a copy as in previous NumPy versions.  In
a future version the read-only restriction will be removed.

Refer to :func:`numpy.diagonal` for full documentation.

See Also
--------
numpy.diagonal : equivalent function
*)

val dot : ?out:Py.Object.t -> b:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.dot(b, out=None)

Dot product of two arrays.

Refer to `numpy.dot` for full documentation.

See Also
--------
numpy.dot : equivalent function

Examples
--------
>>> a = np.eye(2)
>>> b = np.ones((2, 2)) * 2
>>> a.dot(b)
array([[2.,  2.],
       [2.,  2.]])

This array method can be conveniently chained:

>>> a.dot(b).dot(b)
array([[8.,  8.],
       [8.,  8.]])
*)

val dump : file:[`S of string | `Path of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.dump(file)

Dump a pickle of the array to the specified file.
The array can be read back with pickle.load or numpy.load.

Parameters
----------
file : str or Path
    A string naming the dump file.

    .. versionchanged:: 1.17.0
        `pathlib.Path` objects are now accepted.
*)

val dumps : [> tag] Obj.t -> Py.Object.t
(**
a.dumps()

Returns the pickle of the array as a string.
pickle.loads or numpy.loads will convert the string back to an array.

Parameters
----------
None
*)

val fill : value:[`F of float | `I of int | `Bool of bool | `S of string] -> [> tag] Obj.t -> Py.Object.t
(**
a.fill(value)

Fill the array with a scalar value.

Parameters
----------
value : scalar
    All elements of `a` will be assigned this value.

Examples
--------
>>> a = np.array([1, 2])
>>> a.fill(0)
>>> a
array([0, 0])
>>> a = np.empty(2)
>>> a.fill(1)
>>> a
array([1.,  1.])
*)

val flatten : ?order:[`C | `F | `A | `K] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.flatten(order='C')

Return a copy of the array collapsed into one dimension.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    'C' means to flatten in row-major (C-style) order.
    'F' means to flatten in column-major (Fortran-
    style) order. 'A' means to flatten in column-major
    order if `a` is Fortran *contiguous* in memory,
    row-major order otherwise. 'K' means to flatten
    `a` in the order the elements occur in memory.
    The default is 'C'.

Returns
-------
y : ndarray
    A copy of the input array, flattened to one dimension.

See Also
--------
ravel : Return a flattened array.
flat : A 1-D flat iterator over the array.

Examples
--------
>>> a = np.array([[1,2], [3,4]])
>>> a.flatten()
array([1, 2, 3, 4])
>>> a.flatten('F')
array([1, 3, 2, 4])
*)

val getfield : ?offset:int -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.getfield(dtype, offset=0)

Returns a field of the given array as a certain type.

A field is a view of the array data with a given data-type. The values in
the view are determined by the given type and the offset into the current
array in bytes. The offset needs to be such that the view dtype fits in the
array dtype; for example an array of dtype complex128 has 16-byte elements.
If taking a view with a 32-bit integer (4 bytes), the offset needs to be
between 0 and 12 bytes.

Parameters
----------
dtype : str or dtype
    The data type of the view. The dtype size of the view can not be larger
    than that of the array itself.
offset : int
    Number of bytes to skip before beginning the element view.

Examples
--------
>>> x = np.diag([1.+1.j]*2)
>>> x[1, 1] = 2 + 4.j
>>> x
array([[1.+1.j,  0.+0.j],
       [0.+0.j,  2.+4.j]])
>>> x.getfield(np.float64)
array([[1.,  0.],
       [0.,  2.]])

By choosing an offset of 8 bytes we can select the complex part of the
array for our view:

>>> x.getfield(np.float64, offset=8)
array([[1.,  0.],
       [0.,  4.]])
*)

val item : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
a.item( *args)

Copy an element of an array to a standard Python scalar and return it.

Parameters
----------
\*args : Arguments (variable number and type)

    * none: in this case, the method only works for arrays
      with one element (`a.size == 1`), which element is
      copied into a standard Python scalar object and returned.

    * int_type: this argument is interpreted as a flat index into
      the array, specifying which element to copy and return.

    * tuple of int_types: functions as does a single int_type argument,
      except that the argument is interpreted as an nd-index into the
      array.

Returns
-------
z : Standard Python scalar object
    A copy of the specified element of the array as a suitable
    Python scalar

Notes
-----
When the data type of `a` is longdouble or clongdouble, item() returns
a scalar array object because there is no available Python scalar that
would not lose information. Void arrays return a buffer object for item(),
unless fields are defined, in which case a tuple is returned.

`item` is very similar to a[args], except, instead of an array scalar,
a standard Python scalar is returned. This can be useful for speeding up
access to elements of the array and doing arithmetic on elements of the
array using Python's optimized math.

Examples
--------
>>> np.random.seed(123)
>>> x = np.random.randint(9, size=(3, 3))
>>> x
array([[2, 2, 6],
       [1, 3, 6],
       [1, 0, 1]])
>>> x.item(3)
1
>>> x.item(7)
0
>>> x.item((0, 1))
2
>>> x.item((2, 2))
1
*)

val itemset : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
a.itemset( *args)

Insert scalar into an array (scalar is cast to array's dtype, if possible)

There must be at least 1 argument, and define the last argument
as *item*.  Then, ``a.itemset( *args)`` is equivalent to but faster
than ``a[args] = item``.  The item should be a scalar value and `args`
must select a single item in the array `a`.

Parameters
----------
\*args : Arguments
    If one argument: a scalar, only used in case `a` is of size 1.
    If two arguments: the last argument is the value to be set
    and must be a scalar, the first argument specifies a single array
    element location. It is either an int or a tuple.

Notes
-----
Compared to indexing syntax, `itemset` provides some speed increase
for placing a scalar into a particular location in an `ndarray`,
if you must do this.  However, generally this is discouraged:
among other problems, it complicates the appearance of the code.
Also, when using `itemset` (and `item`) inside a loop, be sure
to assign the methods to a local variable to avoid the attribute
look-up at each loop iteration.

Examples
--------
>>> np.random.seed(123)
>>> x = np.random.randint(9, size=(3, 3))
>>> x
array([[2, 2, 6],
       [1, 3, 6],
       [1, 0, 1]])
>>> x.itemset(4, 0)
>>> x.itemset((2, 2), 9)
>>> x
array([[2, 2, 6],
       [1, 0, 6],
       [1, 0, 9]])
*)

val max : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> ?initial:Py.Object.t -> ?where:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

Return the maximum along a given axis.

Refer to `numpy.amax` for full documentation.

See Also
--------
numpy.amax : equivalent function
*)

val mean : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.mean(axis=None, dtype=None, out=None, keepdims=False)

Returns the average of the array elements along given axis.

Refer to `numpy.mean` for full documentation.

See Also
--------
numpy.mean : equivalent function
*)

val min : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> ?initial:Py.Object.t -> ?where:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

Return the minimum along a given axis.

Refer to `numpy.amin` for full documentation.

See Also
--------
numpy.amin : equivalent function
*)

val newbyteorder : ?new_order:string -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
arr.newbyteorder(new_order='S')

Return the array with the same data viewed with a different byte order.

Equivalent to::

    arr.view(arr.dtype.newbytorder(new_order))

Changes are also made in all fields and sub-arrays of the array data
type.



Parameters
----------
new_order : string, optional
    Byte order to force; a value from the byte order specifications
    below. `new_order` codes can be any of:

    * 'S' - swap dtype from current to opposite endian
    * {'<', 'L'} - little endian
    * {'>', 'B'} - big endian
    * {'=', 'N'} - native order
    * {'|', 'I'} - ignore (no change to byte order)

    The default value ('S') results in swapping the current
    byte order. The code does a case-insensitive check on the first
    letter of `new_order` for the alternatives above.  For example,
    any of 'B' or 'b' or 'biggish' are valid to specify big-endian.


Returns
-------
new_arr : array
    New array object with the dtype reflecting given change to the
    byte order.
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
(**
a.nonzero()

Return the indices of the elements that are non-zero.

Refer to `numpy.nonzero` for full documentation.

See Also
--------
numpy.nonzero : equivalent function
*)

val partition : ?axis:int -> ?kind:[`Introselect] -> ?order:[`S of string | `StringList of string list] -> kth:[`I of int | `Is of int list] -> [> tag] Obj.t -> Py.Object.t
(**
a.partition(kth, axis=-1, kind='introselect', order=None)

Rearranges the elements in the array in such a way that the value of the
element in kth position is in the position it would be in a sorted array.
All elements smaller than the kth element are moved before this element and
all equal or greater are moved behind it. The ordering of the elements in
the two partitions is undefined.

.. versionadded:: 1.8.0

Parameters
----------
kth : int or sequence of ints
    Element index to partition by. The kth element value will be in its
    final sorted position and all smaller elements will be moved before it
    and all equal or greater elements behind it.
    The order of all elements in the partitions is undefined.
    If provided with a sequence of kth it will partition all elements
    indexed by kth of them into their sorted position at once.
axis : int, optional
    Axis along which to sort. Default is -1, which means sort along the
    last axis.
kind : {'introselect'}, optional
    Selection algorithm. Default is 'introselect'.
order : str or list of str, optional
    When `a` is an array with fields defined, this argument specifies
    which fields to compare first, second, etc. A single field can
    be specified as a string, and not all fields need to be specified,
    but unspecified fields will still be used, in the order in which
    they come up in the dtype, to break ties.

See Also
--------
numpy.partition : Return a parititioned copy of an array.
argpartition : Indirect partition.
sort : Full sort.

Notes
-----
See ``np.partition`` for notes on the different algorithms.

Examples
--------
>>> a = np.array([3, 4, 2, 1])
>>> a.partition(3)
>>> a
array([2, 1, 3, 4])

>>> a.partition((1, 3))
>>> a
array([1, 2, 3, 4])
*)

val prod : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> ?initial:Py.Object.t -> ?where:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.prod(axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True)

Return the product of the array elements over the given axis

Refer to `numpy.prod` for full documentation.

See Also
--------
numpy.prod : equivalent function
*)

val ptp : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.ptp(axis=None, out=None, keepdims=False)

Peak to peak (maximum - minimum) value along a given axis.

Refer to `numpy.ptp` for full documentation.

See Also
--------
numpy.ptp : equivalent function
*)

val put : ?mode:Py.Object.t -> indices:Py.Object.t -> values:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.put(indices, values, mode='raise')

Set ``a.flat[n] = values[n]`` for all `n` in indices.

Refer to `numpy.put` for full documentation.

See Also
--------
numpy.put : equivalent function
*)

val ravel : ?order:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.ravel([order])

Return a flattened array.

Refer to `numpy.ravel` for full documentation.

See Also
--------
numpy.ravel : equivalent function

ndarray.flat : a flat iterator on the array.
*)

val repeat : ?axis:Py.Object.t -> repeats:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.repeat(repeats, axis=None)

Repeat elements of an array.

Refer to `numpy.repeat` for full documentation.

See Also
--------
numpy.repeat : equivalent function
*)

val reshape : ?order:Py.Object.t -> shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.reshape(shape, order='C')

Returns an array containing the same data with a new shape.

Refer to `numpy.reshape` for full documentation.

See Also
--------
numpy.reshape : equivalent function

Notes
-----
Unlike the free function `numpy.reshape`, this method on `ndarray` allows
the elements of the shape parameter to be passed in as separate arguments.
For example, ``a.reshape(10, 11)`` is equivalent to
``a.reshape((10, 11))``.
*)

val resize : ?refcheck:bool -> new_shape:[`T_n_ints of Py.Object.t | `TupleOfInts of int list] -> [> tag] Obj.t -> Py.Object.t
(**
a.resize(new_shape, refcheck=True)

Change shape and size of array in-place.

Parameters
----------
new_shape : tuple of ints, or `n` ints
    Shape of resized array.
refcheck : bool, optional
    If False, reference count will not be checked. Default is True.

Returns
-------
None

Raises
------
ValueError
    If `a` does not own its own data or references or views to it exist,
    and the data memory must be changed.
    PyPy only: will always raise if the data memory must be changed, since
    there is no reliable way to determine if references or views to it
    exist.

SystemError
    If the `order` keyword argument is specified. This behaviour is a
    bug in NumPy.

See Also
--------
resize : Return a new array with the specified shape.

Notes
-----
This reallocates space for the data area if necessary.

Only contiguous arrays (data elements consecutive in memory) can be
resized.

The purpose of the reference count check is to make sure you
do not use this array as a buffer for another Python object and then
reallocate the memory. However, reference counts can increase in
other ways so if you are sure that you have not shared the memory
for this array with another Python object, then you may safely set
`refcheck` to False.

Examples
--------
Shrinking an array: array is flattened (in the order that the data are
stored in memory), resized, and reshaped:

>>> a = np.array([[0, 1], [2, 3]], order='C')
>>> a.resize((2, 1))
>>> a
array([[0],
       [1]])

>>> a = np.array([[0, 1], [2, 3]], order='F')
>>> a.resize((2, 1))
>>> a
array([[0],
       [2]])

Enlarging an array: as above, but missing entries are filled with zeros:

>>> b = np.array([[0, 1], [2, 3]])
>>> b.resize(2, 3) # new_shape parameter doesn't have to be a tuple
>>> b
array([[0, 1, 2],
       [3, 0, 0]])

Referencing an array prevents resizing...

>>> c = a
>>> a.resize((1, 1))
Traceback (most recent call last):
...
ValueError: cannot resize an array that references or is referenced ...

Unless `refcheck` is False:

>>> a.resize((1, 1), refcheck=False)
>>> a
array([[0]])
>>> c
array([[0]])
*)

val round : ?decimals:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.round(decimals=0, out=None)

Return `a` with each element rounded to the given number of decimals.

Refer to `numpy.around` for full documentation.

See Also
--------
numpy.around : equivalent function
*)

val searchsorted : ?side:Py.Object.t -> ?sorter:Py.Object.t -> v:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.searchsorted(v, side='left', sorter=None)

Find indices where elements of v should be inserted in a to maintain order.

For full documentation, see `numpy.searchsorted`

See Also
--------
numpy.searchsorted : equivalent function
*)

val setfield : ?offset:int -> val_:Py.Object.t -> dtype:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.setfield(val, dtype, offset=0)

Put a value into a specified place in a field defined by a data-type.

Place `val` into `a`'s field defined by `dtype` and beginning `offset`
bytes into the field.

Parameters
----------
val : object
    Value to be placed in field.
dtype : dtype object
    Data-type of the field in which to place `val`.
offset : int, optional
    The number of bytes into the field at which to place `val`.

Returns
-------
None

See Also
--------
getfield

Examples
--------
>>> x = np.eye(3)
>>> x.getfield(np.float64)
array([[1.,  0.,  0.],
       [0.,  1.,  0.],
       [0.,  0.,  1.]])
>>> x.setfield(3, np.int32)
>>> x.getfield(np.int32)
array([[3, 3, 3],
       [3, 3, 3],
       [3, 3, 3]], dtype=int32)
>>> x
array([[1.0e+000, 1.5e-323, 1.5e-323],
       [1.5e-323, 1.0e+000, 1.5e-323],
       [1.5e-323, 1.5e-323, 1.0e+000]])
>>> x.setfield(np.eye(3), np.int32)
>>> x
array([[1.,  0.,  0.],
       [0.,  1.,  0.],
       [0.,  0.,  1.]])
*)

val setflags : ?write:bool -> ?align:bool -> ?uic:bool -> [> tag] Obj.t -> Py.Object.t
(**
a.setflags(write=None, align=None, uic=None)

Set array flags WRITEABLE, ALIGNED, (WRITEBACKIFCOPY and UPDATEIFCOPY),
respectively.

These Boolean-valued flags affect how numpy interprets the memory
area used by `a` (see Notes below). The ALIGNED flag can only
be set to True if the data is actually aligned according to the type.
The WRITEBACKIFCOPY and (deprecated) UPDATEIFCOPY flags can never be set
to True. The flag WRITEABLE can only be set to True if the array owns its
own memory, or the ultimate owner of the memory exposes a writeable buffer
interface, or is a string. (The exception for string is made so that
unpickling can be done without copying memory.)

Parameters
----------
write : bool, optional
    Describes whether or not `a` can be written to.
align : bool, optional
    Describes whether or not `a` is aligned properly for its type.
uic : bool, optional
    Describes whether or not `a` is a copy of another 'base' array.

Notes
-----
Array flags provide information about how the memory area used
for the array is to be interpreted. There are 7 Boolean flags
in use, only four of which can be changed by the user:
WRITEBACKIFCOPY, UPDATEIFCOPY, WRITEABLE, and ALIGNED.

WRITEABLE (W) the data area can be written to;

ALIGNED (A) the data and strides are aligned appropriately for the hardware
(as determined by the compiler);

UPDATEIFCOPY (U) (deprecated), replaced by WRITEBACKIFCOPY;

WRITEBACKIFCOPY (X) this array is a copy of some other array (referenced
by .base). When the C-API function PyArray_ResolveWritebackIfCopy is
called, the base array will be updated with the contents of this array.

All flags can be accessed using the single (upper case) letter as well
as the full name.

Examples
--------
>>> y = np.array([[3, 1, 7],
...               [2, 0, 0],
...               [8, 5, 9]])
>>> y
array([[3, 1, 7],
       [2, 0, 0],
       [8, 5, 9]])
>>> y.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : True
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
>>> y.setflags(write=0, align=0)
>>> y.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : True
  WRITEABLE : False
  ALIGNED : False
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
>>> y.setflags(uic=1)
Traceback (most recent call last):
  File '<stdin>', line 1, in <module>
ValueError: cannot set WRITEBACKIFCOPY flag to True
*)

val sort : ?axis:int -> ?kind:[`Quicksort | `Heapsort | `Stable | `Mergesort] -> ?order:[`S of string | `StringList of string list] -> [> tag] Obj.t -> Py.Object.t
(**
a.sort(axis=-1, kind=None, order=None)

Sort an array in-place. Refer to `numpy.sort` for full documentation.

Parameters
----------
axis : int, optional
    Axis along which to sort. Default is -1, which means sort along the
    last axis.
kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
    Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
    and 'mergesort' use timsort under the covers and, in general, the
    actual implementation will vary with datatype. The 'mergesort' option
    is retained for backwards compatibility.

    .. versionchanged:: 1.15.0.
       The 'stable' option was added.

order : str or list of str, optional
    When `a` is an array with fields defined, this argument specifies
    which fields to compare first, second, etc.  A single field can
    be specified as a string, and not all fields need be specified,
    but unspecified fields will still be used, in the order in which
    they come up in the dtype, to break ties.

See Also
--------
numpy.sort : Return a sorted copy of an array.
numpy.argsort : Indirect sort.
numpy.lexsort : Indirect stable sort on multiple keys.
numpy.searchsorted : Find elements in sorted array.
numpy.partition: Partial sort.

Notes
-----
See `numpy.sort` for notes on the different sorting algorithms.

Examples
--------
>>> a = np.array([[1,4], [3,1]])
>>> a.sort(axis=1)
>>> a
array([[1, 4],
       [1, 3]])
>>> a.sort(axis=0)
>>> a
array([[1, 3],
       [1, 4]])

Use the `order` keyword to specify a field to use when sorting a
structured array:

>>> a = np.array([('a', 2), ('c', 1)], dtype=[('x', 'S1'), ('y', int)])
>>> a.sort(order='y')
>>> a
array([(b'c', 1), (b'a', 2)],
      dtype=[('x', 'S1'), ('y', '<i8')])
*)

val squeeze : ?axis:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.squeeze(axis=None)

Remove single-dimensional entries from the shape of `a`.

Refer to `numpy.squeeze` for full documentation.

See Also
--------
numpy.squeeze : equivalent function
*)

val std : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?ddof:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.std(axis=None, dtype=None, out=None, ddof=0, keepdims=False)

Returns the standard deviation of the array elements along given axis.

Refer to `numpy.std` for full documentation.

See Also
--------
numpy.std : equivalent function
*)

val sum : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> ?initial:Py.Object.t -> ?where:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.sum(axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True)

Return the sum of the array elements over the given axis.

Refer to `numpy.sum` for full documentation.

See Also
--------
numpy.sum : equivalent function
*)

val swapaxes : axis1:Py.Object.t -> axis2:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.swapaxes(axis1, axis2)

Return a view of the array with `axis1` and `axis2` interchanged.

Refer to `numpy.swapaxes` for full documentation.

See Also
--------
numpy.swapaxes : equivalent function
*)

val take : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?mode:Py.Object.t -> indices:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.take(indices, axis=None, out=None, mode='raise')

Return an array formed from the elements of `a` at the given indices.

Refer to `numpy.take` for full documentation.

See Also
--------
numpy.take : equivalent function
*)

val tobytes : ?order:[`C | `F | `None] -> [> tag] Obj.t -> Py.Object.t
(**
a.tobytes(order='C')

Construct Python bytes containing the raw data bytes in the array.

Constructs Python bytes showing a copy of the raw contents of
data memory. The bytes object can be produced in either 'C' or 'Fortran',
or 'Any' order (the default is 'C'-order). 'Any' order means C-order
unless the F_CONTIGUOUS flag in the array is set, in which case it
means 'Fortran' order.

.. versionadded:: 1.9.0

Parameters
----------
order : {'C', 'F', None}, optional
    Order of the data for multidimensional arrays:
    C, Fortran, or the same as for the original array.

Returns
-------
s : bytes
    Python bytes exhibiting a copy of `a`'s raw data.

Examples
--------
>>> x = np.array([[0, 1], [2, 3]], dtype='<u2')
>>> x.tobytes()
b'\x00\x00\x01\x00\x02\x00\x03\x00'
>>> x.tobytes('C') == x.tobytes()
True
>>> x.tobytes('F')
b'\x00\x00\x02\x00\x01\x00\x03\x00'
*)

val tofile : ?sep:string -> ?format:string -> fid:[`S of string | `PyObject of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.tofile(fid, sep='', format='%s')

Write array to a file as text or binary (default).

Data is always written in 'C' order, independent of the order of `a`.
The data produced by this method can be recovered using the function
fromfile().

Parameters
----------
fid : file or str or Path
    An open file object, or a string containing a filename.

    .. versionchanged:: 1.17.0
        `pathlib.Path` objects are now accepted.

sep : str
    Separator between array items for text output.
    If '' (empty), a binary file is written, equivalent to
    ``file.write(a.tobytes())``.
format : str
    Format string for text file output.
    Each entry in the array is formatted to text by first converting
    it to the closest Python type, and then using 'format' % item.

Notes
-----
This is a convenience function for quick storage of array data.
Information on endianness and precision is lost, so this method is not a
good choice for files intended to archive data or transport data between
machines with different endianness. Some of these problems can be overcome
by outputting the data as text files, at the expense of speed and file
size.

When fid is a file object, array contents are directly written to the
file, bypassing the file object's ``write`` method. As a result, tofile
cannot be used with files objects supporting compression (e.g., GzipFile)
or file-like objects that do not support ``fileno()`` (e.g., BytesIO).
*)

val tolist : [> tag] Obj.t -> Py.Object.t
(**
a.tolist()

Return the array as an ``a.ndim``-levels deep nested list of Python scalars.

Return a copy of the array data as a (nested) Python list.
Data items are converted to the nearest compatible builtin Python type, via
the `~numpy.ndarray.item` function.

If ``a.ndim`` is 0, then since the depth of the nested list is 0, it will
not be a list at all, but a simple Python scalar.

Parameters
----------
none

Returns
-------
y : object, or list of object, or list of list of object, or ...
    The possibly nested list of array elements.

Notes
-----
The array may be recreated via ``a = np.array(a.tolist())``, although this
may sometimes lose precision.

Examples
--------
For a 1D array, ``a.tolist()`` is almost the same as ``list(a)``,
except that ``tolist`` changes numpy scalars to Python scalars:

>>> a = np.uint32([1, 2])
>>> a_list = list(a)
>>> a_list
[1, 2]
>>> type(a_list[0])
<class 'numpy.uint32'>
>>> a_tolist = a.tolist()
>>> a_tolist
[1, 2]
>>> type(a_tolist[0])
<class 'int'>

Additionally, for a 2D array, ``tolist`` applies recursively:

>>> a = np.array([[1, 2], [3, 4]])
>>> list(a)
[array([1, 2]), array([3, 4])]
>>> a.tolist()
[[1, 2], [3, 4]]

The base case for this recursion is a 0D array:

>>> a = np.array(1)
>>> list(a)
Traceback (most recent call last):
  ...
TypeError: iteration over a 0-d array
>>> a.tolist()
1
*)

val tostring : ?order:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.tostring(order='C')

A compatibility alias for `tobytes`, with exactly the same behavior.

Despite its name, it returns `bytes` not `str`\ s.

.. deprecated:: 1.19.0
*)

val trace : ?offset:Py.Object.t -> ?axis1:Py.Object.t -> ?axis2:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)

Return the sum along diagonals of the array.

Refer to `numpy.trace` for full documentation.

See Also
--------
numpy.trace : equivalent function
*)

val transpose : Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.transpose( *axes)

Returns a view of the array with axes transposed.

For a 1-D array this has no effect, as a transposed vector is simply the
same vector. To convert a 1-D array into a 2D column vector, an additional
dimension must be added. `np.atleast2d(a).T` achieves this, as does
`a[:, np.newaxis]`.
For a 2-D array, this is a standard matrix transpose.
For an n-D array, if axes are given, their order indicates how the
axes are permuted (see Examples). If axes are not provided and
``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

Parameters
----------
axes : None, tuple of ints, or `n` ints

 * None or no argument: reverses the order of the axes.

 * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
   `i`-th axis becomes `a.transpose()`'s `j`-th axis.

 * `n` ints: same as an n-tuple of the same ints (this form is
   intended simply as a 'convenience' alternative to the tuple form)

Returns
-------
out : ndarray
    View of `a`, with axes suitably permuted.

See Also
--------
ndarray.T : Array property returning the array transposed.
ndarray.reshape : Give a new shape to an array without changing its data.

Examples
--------
>>> a = np.array([[1, 2], [3, 4]])
>>> a
array([[1, 2],
       [3, 4]])
>>> a.transpose()
array([[1, 3],
       [2, 4]])
>>> a.transpose((1, 0))
array([[1, 3],
       [2, 4]])
>>> a.transpose(1, 0)
array([[1, 3],
       [2, 4]])
*)

val var : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?ddof:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.var(axis=None, dtype=None, out=None, ddof=0, keepdims=False)

Returns the variance of the array elements, along given axis.

Refer to `numpy.var` for full documentation.

See Also
--------
numpy.var : equivalent function
*)

val view : ?dtype:[`Ndarray_sub_class of Py.Object.t | `Dtype of Np.Dtype.t] -> ?type_:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.view([dtype][, type])

New view of array with the same data.

.. note::
    Passing None for ``dtype`` is different from omitting the parameter,
    since the former invokes ``dtype(None)`` which is an alias for
    ``dtype('float_')``.

Parameters
----------
dtype : data-type or ndarray sub-class, optional
    Data-type descriptor of the returned view, e.g., float32 or int16.
    Omitting it results in the view having the same data-type as `a`.
    This argument can also be specified as an ndarray sub-class, which
    then specifies the type of the returned object (this is equivalent to
    setting the ``type`` parameter).
type : Python type, optional
    Type of the returned view, e.g., ndarray or matrix.  Again, omission
    of the parameter results in type preservation.

Notes
-----
``a.view()`` is used two different ways:

``a.view(some_dtype)`` or ``a.view(dtype=some_dtype)`` constructs a view
of the array's memory with a different data-type.  This can cause a
reinterpretation of the bytes of memory.

``a.view(ndarray_subclass)`` or ``a.view(type=ndarray_subclass)`` just
returns an instance of `ndarray_subclass` that looks at the same array
(same shape, dtype, etc.)  This does not cause a reinterpretation of the
memory.

For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of
bytes per entry than the previous dtype (for example, converting a
regular array to a structured array), then the behavior of the view
cannot be predicted just from the superficial appearance of ``a`` (shown
by ``print(a)``). It also depends on exactly how ``a`` is stored in
memory. Therefore if ``a`` is C-ordered versus fortran-ordered, versus
defined as a slice or transpose, etc., the view may give different
results.


Examples
--------
>>> x = np.array([(1, 2)], dtype=[('a', np.int8), ('b', np.int8)])

Viewing array data using a different type and dtype:

>>> y = x.view(dtype=np.int16, type=np.matrix)
>>> y
matrix([[513]], dtype=int16)
>>> print(type(y))
<class 'numpy.matrix'>

Creating a view on a structured array so it can be used in calculations

>>> x = np.array([(1, 2),(3,4)], dtype=[('a', np.int8), ('b', np.int8)])
>>> xv = x.view(dtype=np.int8).reshape(-1,2)
>>> xv
array([[1, 2],
       [3, 4]], dtype=int8)
>>> xv.mean(0)
array([2.,  3.])

Making changes to the view changes the underlying array

>>> xv[0,1] = 20
>>> x
array([(1, 20), (3,  4)], dtype=[('a', 'i1'), ('b', 'i1')])

Using a view to convert an array to a recarray:

>>> z = x.view(np.recarray)
>>> z.a
array([1, 3], dtype=int8)

Views share data:

>>> x[0] = (9, 10)
>>> z[0]
(9, 10)

Views that change the dtype size (bytes per entry) should normally be
avoided on arrays defined by slices, transposes, fortran-ordering, etc.:

>>> x = np.array([[1,2,3],[4,5,6]], dtype=np.int16)
>>> y = x[:, 0:2]
>>> y
array([[1, 2],
       [4, 5]], dtype=int16)
>>> y.view(dtype=[('width', np.int16), ('length', np.int16)])
Traceback (most recent call last):
    ...
ValueError: To change to a dtype of a different size, the array must be C-contiguous
>>> z = y.copy()
>>> z.view(dtype=[('width', np.int16), ('length', np.int16)])
array([[(1, 2)],
       [(4, 5)]], dtype=[('width', '<i2'), ('length', '<i2')])
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val convert_dtypes : dtype_template:Py.Object.t -> order_code:string -> unit -> Py.Object.t
(**
Convert dtypes in mapping to given order

Parameters
----------
dtype_template : mapping
   mapping with values returning numpy dtype from ``np.dtype(val)``
order_code : str
   an order code suitable for using in ``dtype.newbyteorder()``

Returns
-------
dtypes : mapping
   mapping where values have been replaced by
   ``np.dtype(val).newbyteorder(order_code)``
*)


end

module Mio5_utils : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module VarHeader5 : sig
type tag = [`VarHeader5]
type t = [`Object | `VarHeader5] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Csc_matrix : sig
type tag = [`Csc_matrix]
type t = [`ArrayLike | `Csc_matrix | `IndexMixin | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val as_index : t -> [`IndexMixin] Obj.t
val create : ?shape:Py.Object.t -> ?dtype:Py.Object.t -> ?copy:Py.Object.t -> arg1:Py.Object.t -> unit -> t
(**
Compressed Sparse Column matrix

This can be instantiated in several ways:

    csc_matrix(D)
        with a dense matrix or rank-2 ndarray D

    csc_matrix(S)
        with another sparse matrix S (equivalent to S.tocsc())

    csc_matrix((M, N), [dtype])
        to construct an empty matrix with shape (M, N)
        dtype is optional, defaulting to dtype='d'.

    csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
        where ``data``, ``row_ind`` and ``col_ind`` satisfy the
        relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

    csc_matrix((data, indices, indptr), [shape=(M, N)])
        is the standard CSC representation where the row indices for
        column i are stored in ``indices[indptr[i]:indptr[i+1]]``
        and their corresponding values are stored in
        ``data[indptr[i]:indptr[i+1]]``.  If the shape parameter is
        not supplied, the matrix dimensions are inferred from
        the index arrays.

Attributes
----------
dtype : dtype
    Data type of the matrix
shape : 2-tuple
    Shape of the matrix
ndim : int
    Number of dimensions (this is always 2)
nnz
    Number of stored values, including explicit zeros
data
    Data array of the matrix
indices
    CSC format index array
indptr
    CSC format index pointer array
has_sorted_indices
    Whether indices are sorted

Notes
-----

Sparse matrices can be used in arithmetic operations: they support
addition, subtraction, multiplication, division, and matrix power.

Advantages of the CSC format
    - efficient arithmetic operations CSC + CSC, CSC * CSC, etc.
    - efficient column slicing
    - fast matrix vector products (CSR, BSR may be faster)

Disadvantages of the CSC format
  - slow row slicing operations (consider CSR)
  - changes to the sparsity structure are expensive (consider LIL or DOK)


Examples
--------

>>> import numpy as np
>>> from scipy.sparse import csc_matrix
>>> csc_matrix((3, 4), dtype=np.int8).toarray()
array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]], dtype=int8)

>>> row = np.array([0, 2, 2, 0, 1, 2])
>>> col = np.array([0, 0, 1, 2, 2, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> csc_matrix((data, (row, col)), shape=(3, 3)).toarray()
array([[1, 0, 4],
       [0, 0, 5],
       [2, 3, 6]])

>>> indptr = np.array([0, 2, 3, 6])
>>> indices = np.array([0, 2, 2, 0, 1, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> csc_matrix((data, indices, indptr), shape=(3, 3)).toarray()
array([[1, 0, 4],
       [0, 0, 5],
       [2, 3, 6]])
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val __setitem__ : key:Py.Object.t -> x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
None
*)

val arcsin : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arcsin.

See `numpy.arcsin` for more information.
*)

val arcsinh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arcsinh.

See `numpy.arcsinh` for more information.
*)

val arctan : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arctan.

See `numpy.arctan` for more information.
*)

val arctanh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arctanh.

See `numpy.arctanh` for more information.
*)

val argmax : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return indices of maximum elements along an axis.

Implicit zero elements are also taken into account. If there are
several maximum values, the index of the first occurrence is returned.

Parameters
----------
axis : {-2, -1, 0, 1, None}, optional
    Axis along which the argmax is computed. If None (default), index
    of the maximum element in the flatten data is returned.
out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
ind : numpy.matrix or int
    Indices of maximum elements. If matrix, its size along `axis` is 1.
*)

val argmin : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return indices of minimum elements along an axis.

Implicit zero elements are also taken into account. If there are
several minimum values, the index of the first occurrence is returned.

Parameters
----------
axis : {-2, -1, 0, 1, None}, optional
    Axis along which the argmin is computed. If None (default), index
    of the minimum element in the flatten data is returned.
out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
 ind : numpy.matrix or int
    Indices of minimum elements. If matrix, its size along `axis` is 1.
*)

val asformat : ?copy:bool -> format:[`S of string | `None] -> [> tag] Obj.t -> Py.Object.t
(**
Return this matrix in the passed format.

Parameters
----------
format : {str, None}
    The desired matrix format ('csr', 'csc', 'lil', 'dok', 'array', ...)
    or None for no conversion.
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : This matrix in the passed format.
*)

val asfptype : [> tag] Obj.t -> Py.Object.t
(**
Upcast matrix to a floating point format (if necessary)
*)

val astype : ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
(**
Cast the matrix elements to a specified type.

Parameters
----------
dtype : string or numpy dtype
    Typecode or data-type to which to cast the data.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur.
    Defaults to 'unsafe' for backwards compatibility.
    'no' means the data types should not be cast at all.
    'equiv' means only byte-order changes are allowed.
    'safe' means only casts which can preserve values are allowed.
    'same_kind' means only safe casts or casts within a kind,
    like float64 to float32, are allowed.
    'unsafe' means any data conversions may be done.
copy : bool, optional
    If `copy` is `False`, the result might share some memory with this
    matrix. If `copy` is `True`, it is guaranteed that the result and
    this matrix do not share any memory.
*)

val ceil : [> tag] Obj.t -> Py.Object.t
(**
Element-wise ceil.

See `numpy.ceil` for more information.
*)

val check_format : ?full_check:bool -> [> tag] Obj.t -> Py.Object.t
(**
check whether the matrix format is valid

Parameters
----------
full_check : bool, optional
    If `True`, rigorous check, O(N) operations. Otherwise
    basic check, O(1) operations (default True).
*)

val conj : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise complex conjugation.

If the matrix is of non-complex data type and `copy` is False,
this method does nothing and the data is not copied.

Parameters
----------
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : The element-wise complex conjugate.
*)

val conjugate : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise complex conjugation.

If the matrix is of non-complex data type and `copy` is False,
this method does nothing and the data is not copied.

Parameters
----------
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : The element-wise complex conjugate.
*)

val copy : [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of this matrix.

No data/indices will be shared between the returned value and current
matrix.
*)

val count_nonzero : [> tag] Obj.t -> Py.Object.t
(**
Number of non-zero entries, equivalent to

np.count_nonzero(a.toarray())

Unlike getnnz() and the nnz property, which return the number of stored
entries (the length of the data attribute), this method counts the
actual number of non-zero entries in data.
*)

val deg2rad : [> tag] Obj.t -> Py.Object.t
(**
Element-wise deg2rad.

See `numpy.deg2rad` for more information.
*)

val diagonal : ?k:int -> [> tag] Obj.t -> Py.Object.t
(**
Returns the kth diagonal of the matrix.

Parameters
----------
k : int, optional
    Which diagonal to get, corresponding to elements a[i, i+k].
    Default: 0 (the main diagonal).

    .. versionadded:: 1.0

See also
--------
numpy.diagonal : Equivalent numpy function.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> A.diagonal()
array([1, 0, 5])
>>> A.diagonal(k=1)
array([2, 3])
*)

val dot : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Ordinary dot product

Examples
--------
>>> import numpy as np
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> v = np.array([1, 0, -1])
>>> A.dot(v)
array([ 1, -3, -1], dtype=int64)
*)

val eliminate_zeros : [> tag] Obj.t -> Py.Object.t
(**
Remove zero entries from the matrix

This is an *in place* operation
*)

val expm1 : [> tag] Obj.t -> Py.Object.t
(**
Element-wise expm1.

See `numpy.expm1` for more information.
*)

val floor : [> tag] Obj.t -> Py.Object.t
(**
Element-wise floor.

See `numpy.floor` for more information.
*)

val getH : [> tag] Obj.t -> Py.Object.t
(**
Return the Hermitian transpose of this matrix.

See Also
--------
numpy.matrix.getH : NumPy's implementation of `getH` for matrices
*)

val get_shape : [> tag] Obj.t -> Py.Object.t
(**
Get shape of a matrix.
*)

val getcol : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of column i of the matrix, as a (m x 1)
CSC matrix (column vector).
*)

val getformat : [> tag] Obj.t -> Py.Object.t
(**
Format of a matrix representation as a string.
*)

val getmaxprint : [> tag] Obj.t -> Py.Object.t
(**
Maximum number of elements to display when printed.
*)

val getnnz : ?axis:[`One | `Zero] -> [> tag] Obj.t -> Py.Object.t
(**
Number of stored values, including explicit zeros.

Parameters
----------
axis : None, 0, or 1
    Select between the number of values across the whole matrix, in
    each column, or in each row.

See also
--------
count_nonzero : Number of non-zero entries
*)

val getrow : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of row i of the matrix, as a (1 x n)
CSR matrix (row vector).
*)

val log1p : [> tag] Obj.t -> Py.Object.t
(**
Element-wise log1p.

See `numpy.log1p` for more information.
*)

val max : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the maximum of the matrix or maximum along an axis.
This takes all elements into account, not just the non-zero ones.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the maximum over all the matrix elements, returning
    a scalar (i.e., `axis` = `None`).

out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except
    for the default value, as this argument is not used.

Returns
-------
amax : coo_matrix or scalar
    Maximum of `a`. If `axis` is None, the result is a scalar value.
    If `axis` is given, the result is a sparse.coo_matrix of dimension
    ``a.ndim - 1``.

See Also
--------
min : The minimum value of a sparse matrix along a given axis.
numpy.matrix.max : NumPy's implementation of 'max' for matrices
*)

val maximum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise maximum between this and another matrix.
*)

val mean : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the arithmetic mean along the specified axis.

Returns the average of the matrix elements. The average is taken
over all elements in the matrix by default, otherwise over the
specified axis. `float64` intermediate and return values are used
for integer inputs.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the mean is computed. The default is to compute
    the mean of all elements in the matrix (i.e., `axis` = `None`).
dtype : data-type, optional
    Type to use in computing the mean. For integer inputs, the default
    is `float64`; for floating point inputs, it is the same as the
    input dtype.

    .. versionadded:: 0.18.0

out : np.matrix, optional
    Alternative output matrix in which to place the result. It must
    have the same shape as the expected output, but the type of the
    output values will be cast if necessary.

    .. versionadded:: 0.18.0

Returns
-------
m : np.matrix

See Also
--------
numpy.matrix.mean : NumPy's implementation of 'mean' for matrices
*)

val min : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the minimum of the matrix or maximum along an axis.
This takes all elements into account, not just the non-zero ones.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the minimum over all the matrix elements, returning
    a scalar (i.e., `axis` = `None`).

out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
amin : coo_matrix or scalar
    Minimum of `a`. If `axis` is None, the result is a scalar value.
    If `axis` is given, the result is a sparse.coo_matrix of dimension
    ``a.ndim - 1``.

See Also
--------
max : The maximum value of a sparse matrix along a given axis.
numpy.matrix.min : NumPy's implementation of 'min' for matrices
*)

val minimum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise minimum between this and another matrix.
*)

val multiply : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Point-wise multiplication by another matrix, vector, or
scalar.
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
(**
nonzero indices

Returns a tuple of arrays (row,col) containing the indices
of the non-zero elements of the matrix.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1,2,0],[0,0,3],[4,0,5]])
>>> A.nonzero()
(array([0, 0, 1, 2, 2]), array([0, 1, 2, 0, 2]))
*)

val power : ?dtype:Py.Object.t -> n:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
This function performs element-wise power.

Parameters
----------
n : n is a scalar

dtype : If dtype is not specified, the current dtype will be preserved.
*)

val prune : [> tag] Obj.t -> Py.Object.t
(**
Remove empty space after all non-zero elements.
        
*)

val rad2deg : [> tag] Obj.t -> Py.Object.t
(**
Element-wise rad2deg.

See `numpy.rad2deg` for more information.
*)

val reshape : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
reshape(self, shape, order='C', copy=False)

Gives a new shape to a sparse matrix without changing its data.

Parameters
----------
shape : length-2 tuple of ints
    The new shape should be compatible with the original shape.
order : {'C', 'F'}, optional
    Read the elements using this index order. 'C' means to read and
    write the elements using C-like index order; e.g., read entire first
    row, then second row, etc. 'F' means to read and write the elements
    using Fortran-like index order; e.g., read entire first column, then
    second column, etc.
copy : bool, optional
    Indicates whether or not attributes of self should be copied
    whenever possible. The degree to which attributes are copied varies
    depending on the type of sparse matrix being used.

Returns
-------
reshaped_matrix : sparse matrix
    A sparse matrix with the given `shape`, not necessarily of the same
    format as the current object.

See Also
--------
numpy.matrix.reshape : NumPy's implementation of 'reshape' for
                       matrices
*)

val resize : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
Resize the matrix in-place to dimensions given by ``shape``

Any elements that lie within the new shape will remain at the same
indices, while non-zero elements lying outside the new shape are
removed.

Parameters
----------
shape : (int, int)
    number of rows and columns in the new matrix

Notes
-----
The semantics are not identical to `numpy.ndarray.resize` or
`numpy.resize`. Here, the same data will be maintained at each index
before and after reshape, if that index is within the new bounds. In
numpy, resizing maintains contiguity of the array, moving elements
around in the logical matrix but not within a flattened representation.

We give no guarantees about whether the underlying data attributes
(arrays, etc.) will be modified in place or replaced with new objects.
*)

val rint : [> tag] Obj.t -> Py.Object.t
(**
Element-wise rint.

See `numpy.rint` for more information.
*)

val set_shape : shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
See `reshape`.
*)

val setdiag : ?k:int -> values:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Set diagonal or off-diagonal elements of the array.

Parameters
----------
values : array_like
    New values of the diagonal elements.

    Values may have any length. If the diagonal is longer than values,
    then the remaining diagonal entries will not be set. If values if
    longer than the diagonal, then the remaining values are ignored.

    If a scalar value is given, all of the diagonal is set to it.

k : int, optional
    Which off-diagonal to set, corresponding to elements a[i,i+k].
    Default: 0 (the main diagonal).
*)

val sign : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sign.

See `numpy.sign` for more information.
*)

val sin : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sin.

See `numpy.sin` for more information.
*)

val sinh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sinh.

See `numpy.sinh` for more information.
*)

val sort_indices : [> tag] Obj.t -> Py.Object.t
(**
Sort the indices of this matrix *in place*
        
*)

val sorted_indices : [> tag] Obj.t -> Py.Object.t
(**
Return a copy of this matrix with sorted indices
        
*)

val sqrt : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sqrt.

See `numpy.sqrt` for more information.
*)

val sum : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Sum the matrix elements over a given axis.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the sum of all the matrix elements, returning a scalar
    (i.e., `axis` = `None`).
dtype : dtype, optional
    The type of the returned matrix and of the accumulator in which
    the elements are summed.  The dtype of `a` is used by default
    unless `a` has an integer dtype of less precision than the default
    platform integer.  In that case, if `a` is signed then the platform
    integer is used while if `a` is unsigned then an unsigned integer
    of the same precision as the platform integer is used.

    .. versionadded:: 0.18.0

out : np.matrix, optional
    Alternative output matrix in which to place the result. It must
    have the same shape as the expected output, but the type of the
    output values will be cast if necessary.

    .. versionadded:: 0.18.0

Returns
-------
sum_along_axis : np.matrix
    A matrix with the same shape as `self`, with the specified
    axis removed.

See Also
--------
numpy.matrix.sum : NumPy's implementation of 'sum' for matrices
*)

val sum_duplicates : [> tag] Obj.t -> Py.Object.t
(**
Eliminate duplicate matrix entries by adding them together

The is an *in place* operation
*)

val tan : [> tag] Obj.t -> Py.Object.t
(**
Element-wise tan.

See `numpy.tan` for more information.
*)

val tanh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise tanh.

See `numpy.tanh` for more information.
*)

val toarray : ?order:[`C | `F] -> ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `T2_D of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
Return a dense ndarray representation of this matrix.

Parameters
----------
order : {'C', 'F'}, optional
    Whether to store multidimensional data in C (row-major)
    or Fortran (column-major) order in memory. The default
    is 'None', indicating the NumPy default of C-ordered.
    Cannot be specified in conjunction with the `out`
    argument.

out : ndarray, 2-D, optional
    If specified, uses this array as the output buffer
    instead of allocating a new array to return. The provided
    array must have the same shape and dtype as the sparse
    matrix on which you are calling the method. For most
    sparse types, `out` is required to be memory contiguous
    (either C or Fortran ordered).

Returns
-------
arr : ndarray, 2-D
    An array with the same shape and containing the same
    data represented by the sparse matrix, with the requested
    memory order. If `out` was passed, the same object is
    returned after being modified in-place to contain the
    appropriate values.
*)

val tobsr : ?blocksize:Py.Object.t -> ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Block Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant bsr_matrix.

When blocksize=(R, C) is provided, it will be used for construction of
the bsr_matrix.
*)

val tocoo : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to COOrdinate format.

With copy=False, the data/indices may be shared between this matrix and
the resultant coo_matrix.
*)

val tocsc : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Column format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csc_matrix.
*)

val tocsr : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant csr_matrix.
*)

val todense : ?order:[`C | `F] -> ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `T2_D of Py.Object.t] -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return a dense matrix representation of this matrix.

Parameters
----------
order : {'C', 'F'}, optional
    Whether to store multi-dimensional data in C (row-major)
    or Fortran (column-major) order in memory. The default
    is 'None', indicating the NumPy default of C-ordered.
    Cannot be specified in conjunction with the `out`
    argument.

out : ndarray, 2-D, optional
    If specified, uses this array (or `numpy.matrix`) as the
    output buffer instead of allocating a new array to
    return. The provided array must have the same shape and
    dtype as the sparse matrix on which you are calling the
    method.

Returns
-------
arr : numpy.matrix, 2-D
    A NumPy matrix object with the same shape and containing
    the same data represented by the sparse matrix, with the
    requested memory order. If `out` was passed and was an
    array (rather than a `numpy.matrix`), it will be filled
    with the appropriate values and returned wrapped in a
    `numpy.matrix` object that shares the same memory.
*)

val todia : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to sparse DIAgonal format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dia_matrix.
*)

val todok : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Dictionary Of Keys format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dok_matrix.
*)

val tolil : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to List of Lists format.

With copy=False, the data/indices may be shared between this matrix and
the resultant lil_matrix.
*)

val transpose : ?axes:Py.Object.t -> ?copy:bool -> [> tag] Obj.t -> Py.Object.t
(**
Reverses the dimensions of the sparse matrix.

Parameters
----------
axes : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except
    for the default value.
copy : bool, optional
    Indicates whether or not attributes of `self` should be
    copied whenever possible. The degree to which attributes
    are copied varies depending on the type of sparse matrix
    being used.

Returns
-------
p : `self` with the dimensions reversed.

See Also
--------
numpy.matrix.transpose : NumPy's implementation of 'transpose'
                         for matrices
*)

val trunc : [> tag] Obj.t -> Py.Object.t
(**
Element-wise trunc.

See `numpy.trunc` for more information.
*)


(** Attribute dtype: get value or raise Not_found if None.*)
val dtype : t -> Np.Dtype.t

(** Attribute dtype: get value as an option. *)
val dtype_opt : t -> (Np.Dtype.t) option


(** Attribute shape: get value or raise Not_found if None.*)
val shape : t -> Py.Object.t

(** Attribute shape: get value as an option. *)
val shape_opt : t -> (Py.Object.t) option


(** Attribute ndim: get value or raise Not_found if None.*)
val ndim : t -> int

(** Attribute ndim: get value as an option. *)
val ndim_opt : t -> (int) option


(** Attribute nnz: get value or raise Not_found if None.*)
val nnz : t -> Py.Object.t

(** Attribute nnz: get value as an option. *)
val nnz_opt : t -> (Py.Object.t) option


(** Attribute data: get value or raise Not_found if None.*)
val data : t -> Py.Object.t

(** Attribute data: get value as an option. *)
val data_opt : t -> (Py.Object.t) option


(** Attribute indices: get value or raise Not_found if None.*)
val indices : t -> Py.Object.t

(** Attribute indices: get value as an option. *)
val indices_opt : t -> (Py.Object.t) option


(** Attribute indptr: get value or raise Not_found if None.*)
val indptr : t -> Py.Object.t

(** Attribute indptr: get value as an option. *)
val indptr_opt : t -> (Py.Object.t) option


(** Attribute has_sorted_indices: get value or raise Not_found if None.*)
val has_sorted_indices : t -> Py.Object.t

(** Attribute has_sorted_indices: get value as an option. *)
val has_sorted_indices_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val asbytes : Py.Object.t -> Py.Object.t
(**
None
*)

val asstr : Py.Object.t -> Py.Object.t
(**
None
*)

val pycopy : Py.Object.t -> Py.Object.t
(**
Shallow copy operation on arbitrary Python objects.

See the module's __doc__ string for more info.
*)


end

module Mio_utils : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t


end

module Miobase : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module MatVarReader : sig
type tag = [`MatVarReader]
type t = [`MatVarReader | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : Py.Object.t -> t
(**
Abstract class defining required interface for var readers
*)

val array_from_header : header:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Reads array given header 
*)

val read_header : [> tag] Obj.t -> Py.Object.t
(**
Returns header 
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val arr_dtype_number : arr:Py.Object.t -> num:Py.Object.t -> unit -> Py.Object.t
(**
Return dtype for given number of items per element
*)

val arr_to_chars : Py.Object.t -> Py.Object.t
(**
Convert string array to char array 
*)

val convert_dtypes : dtype_template:Py.Object.t -> order_code:string -> unit -> Py.Object.t
(**
Convert dtypes in mapping to given order

Parameters
----------
dtype_template : mapping
   mapping with values returning numpy dtype from ``np.dtype(val)``
order_code : str
   an order code suitable for using in ``dtype.newbyteorder()``

Returns
-------
dtypes : mapping
   mapping where values have been replaced by
   ``np.dtype(val).newbyteorder(order_code)``
*)

val docfiller : Py.Object.t -> Py.Object.t
(**
None
*)

val get_matfile_version : Py.Object.t -> (Py.Object.t * int)
(**
Return major, minor tuple depending on apparent mat file type

Where:

 #. 0,x -> version 4 format mat files
 #. 1,x -> version 5 format mat files
 #. 2,x -> version 7.3 format mat files (HDF format)

Parameters
----------
fileobj : file_like
    object implementing seek() and read()

Returns
-------
major_version : {0, 1, 2}
    major MATLAB File format version
minor_version : int
    minor MATLAB file format version

Raises
------
MatReadError
    If the file is empty.
ValueError
    The matfile version is unknown.

Notes
-----
Has the side effect of setting the file read pointer to 0
*)

val matdims : ?oned_as:[`Column | `Row] -> arr:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Determine equivalent MATLAB dimensions for given array

Parameters
----------
arr : ndarray
    Input array
oned_as : {'column', 'row'}, optional
    Whether 1-D arrays are returned as MATLAB row or column matrices.
    Default is 'column'.

Returns
-------
dims : tuple
    Shape tuple, in the form MATLAB expects it.

Notes
-----
We had to decide what shape a 1 dimensional array would be by
default. ``np.atleast_2d`` thinks it is a row vector. The
default for a vector in MATLAB (e.g., ``>> 1:12``) is a row vector.

Versions of scipy up to and including 0.11 resulted (accidentally)
in 1-D arrays being read as column vectors. For the moment, we
maintain the same tradition here.

Examples
--------
>>> matdims(np.array(1)) # NumPy scalar
(1, 1)
>>> matdims(np.array([1])) # 1-D array, 1 element
(1, 1)
>>> matdims(np.array([1,2])) # 1-D array, 2 elements
(2, 1)
>>> matdims(np.array([[2],[3]])) # 2-D array, column vector
(2, 1)
>>> matdims(np.array([[2,3]])) # 2-D array, row vector
(1, 2)
>>> matdims(np.array([[[2,3]]])) # 3-D array, rowish vector
(1, 1, 2)
>>> matdims(np.array([])) # empty 1-D array
(0, 0)
>>> matdims(np.array([[]])) # empty 2-D array
(0, 0)
>>> matdims(np.array([[[]]])) # empty 3-D array
(0, 0, 0)

Optional argument flips 1-D shape behavior.

>>> matdims(np.array([1,2]), 'row') # 1-D array, 2 elements
(1, 2)

The argument has to make sense though

>>> matdims(np.array([1,2]), 'bizarre')
Traceback (most recent call last):
   ...
ValueError: 1-D option 'bizarre' is strange
*)

val read_dtype : mat_stream:Py.Object.t -> a_dtype:Np.Dtype.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Generic get of byte stream data of known type

Parameters
----------
mat_stream : file_like object
    MATLAB (tm) mat file stream
a_dtype : dtype
    dtype of array to read. `a_dtype` is assumed to be correct
    endianness.

Returns
-------
arr : ndarray
    Array of dtype `a_dtype` read from stream.
*)


end

module Streams : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module GenericStream : sig
type tag = [`GenericStream]
type t = [`GenericStream | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end


end

val loadmat : ?mdict:Py.Object.t -> ?appendmat:bool -> ?kwargs:(string * Py.Object.t) list -> file_name:string -> unit -> Py.Object.t
(**
Load MATLAB file.

Parameters
----------
file_name : str
   Name of the mat file (do not need .mat extension if
   appendmat==True). Can also pass open file-like object.
mdict : dict, optional
    Dictionary in which to insert matfile variables.
appendmat : bool, optional
   True to append the .mat extension to the end of the given
   filename, if not already present.
byte_order : str or None, optional
   None by default, implying byte order guessed from mat
   file. Otherwise can be one of ('native', '=', 'little', '<',
   'BIG', '>').
mat_dtype : bool, optional
   If True, return arrays in same dtype as would be loaded into
   MATLAB (instead of the dtype with which they are saved).
squeeze_me : bool, optional
   Whether to squeeze unit matrix dimensions or not.
chars_as_strings : bool, optional
   Whether to convert char arrays to string arrays.
matlab_compatible : bool, optional
   Returns matrices as would be loaded by MATLAB (implies
   squeeze_me=False, chars_as_strings=False, mat_dtype=True,
   struct_as_record=True).
struct_as_record : bool, optional
   Whether to load MATLAB structs as NumPy record arrays, or as
   old-style NumPy arrays with dtype=object. Setting this flag to
   False replicates the behavior of scipy version 0.7.x (returning
   NumPy object arrays). The default setting is True, because it
   allows easier round-trip load and save of MATLAB files.
verify_compressed_data_integrity : bool, optional
    Whether the length of compressed sequences in the MATLAB file
    should be checked, to ensure that they are not longer than we expect.
    It is advisable to enable this (the default) because overlong
    compressed sequences in MATLAB files generally indicate that the
    files have experienced some sort of corruption.
variable_names : None or sequence
    If None (the default) - read all variables in file. Otherwise,
    `variable_names` should be a sequence of strings, giving names of the
    MATLAB variables to read from the file. The reader will skip any
    variable with a name not in this sequence, possibly saving some read
    processing.
simplify_cells : False, optional
    If True, return a simplified dict structure (which is useful if the mat
    file contains cell arrays). Note that this only affects the structure
    of the result and not its contents (which is identical for both output
    structures). If True, this automatically sets `struct_as_record` to
    False and `squeeze_me` to True, which is required to simplify cells.

Returns
-------
mat_dict : dict
   dictionary with variable names as keys, and loaded matrices as
   values.

Notes
-----
v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.

You will need an HDF5 Python library to read MATLAB 7.3 format mat
files. Because SciPy does not supply one, we do not implement the
HDF5 / 7.3 interface here.

Examples
--------
>>> from os.path import dirname, join as pjoin
>>> import scipy.io as sio

Get the filename for an example .mat file from the tests/data directory.

>>> data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
>>> mat_fname = pjoin(data_dir, 'testdouble_7.4_GLNX86.mat')

Load the .mat file contents.

>>> mat_contents = sio.loadmat(mat_fname)

The result is a dictionary, one key/value pair for each variable:

>>> sorted(mat_contents.keys())
['__globals__', '__header__', '__version__', 'testdouble']
>>> mat_contents['testdouble']
array([[0.        , 0.78539816, 1.57079633, 2.35619449, 3.14159265,
        3.92699082, 4.71238898, 5.49778714, 6.28318531]])

By default SciPy reads MATLAB structs as structured NumPy arrays where the
dtype fields are of type `object` and the names correspond to the MATLAB
struct field names. This can be disabled by setting the optional argument
`struct_as_record=False`.

Get the filename for an example .mat file that contains a MATLAB struct
called `teststruct` and load the contents.

>>> matstruct_fname = pjoin(data_dir, 'teststruct_7.4_GLNX86.mat')
>>> matstruct_contents = sio.loadmat(matstruct_fname)
>>> teststruct = matstruct_contents['teststruct']
>>> teststruct.dtype
dtype([('stringfield', 'O'), ('doublefield', 'O'), ('complexfield', 'O')])

The size of the structured array is the size of the MATLAB struct, not the
number of elements in any particular field. The shape defaults to 2-D
unless the optional argument `squeeze_me=True`, in which case all length 1
dimensions are removed.

>>> teststruct.size
1
>>> teststruct.shape
(1, 1)

Get the 'stringfield' of the first element in the MATLAB struct.

>>> teststruct[0, 0]['stringfield']
array(['Rats live on no evil star.'],
  dtype='<U26')

Get the first element of the 'doublefield'.

>>> teststruct['doublefield'][0, 0]
array([[ 1.41421356,  2.71828183,  3.14159265]])

Load the MATLAB struct, squeezing out length 1 dimensions, and get the item
from the 'complexfield'.

>>> matstruct_squeezed = sio.loadmat(matstruct_fname, squeeze_me=True)
>>> matstruct_squeezed['teststruct'].shape
()
>>> matstruct_squeezed['teststruct']['complexfield'].shape
()
>>> matstruct_squeezed['teststruct']['complexfield'].item()
array([ 1.41421356+1.41421356j,  2.71828183+2.71828183j,
    3.14159265+3.14159265j])
*)

val savemat : ?appendmat:bool -> ?format:[`T4 | `T5] -> ?long_field_names:bool -> ?do_compression:bool -> ?oned_as:[`Row | `Column] -> file_name:[`S of string | `File_like_object of Py.Object.t] -> mdict:Py.Object.t -> unit -> Py.Object.t
(**
Save a dictionary of names and arrays into a MATLAB-style .mat file.

This saves the array objects in the given dictionary to a MATLAB-
style .mat file.

Parameters
----------
file_name : str or file-like object
    Name of the .mat file (.mat extension not needed if ``appendmat ==
    True``).
    Can also pass open file_like object.
mdict : dict
    Dictionary from which to save matfile variables.
appendmat : bool, optional
    True (the default) to append the .mat extension to the end of the
    given filename, if not already present.
format : {'5', '4'}, string, optional
    '5' (the default) for MATLAB 5 and up (to 7.2),
    '4' for MATLAB 4 .mat files.
long_field_names : bool, optional
    False (the default) - maximum field name length in a structure is
    31 characters which is the documented maximum length.
    True - maximum field name length in a structure is 63 characters
    which works for MATLAB 7.6+.
do_compression : bool, optional
    Whether or not to compress matrices on write. Default is False.
oned_as : {'row', 'column'}, optional
    If 'column', write 1-D NumPy arrays as column vectors.
    If 'row', write 1-D NumPy arrays as row vectors.

Examples
--------
>>> from scipy.io import savemat
>>> a = np.arange(20)
>>> mdic = {'a': a, 'label': 'experiment'}
>>> mdic
{'a': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19]),
'label': 'experiment'}
>>> savemat('matlab_matrix.mat', mdic)
*)

val whosmat : ?appendmat:bool -> ?kwargs:(string * Py.Object.t) list -> file_name:string -> unit -> Py.Object.t
(**
List variables inside a MATLAB file.

Parameters
----------
file_name : str
   Name of the mat file (do not need .mat extension if
   appendmat==True) Can also pass open file-like object.
appendmat : bool, optional
   True to append the .mat extension to the end of the given
   filename, if not already present.
byte_order : str or None, optional
   None by default, implying byte order guessed from mat
   file. Otherwise can be one of ('native', '=', 'little', '<',
   'BIG', '>').
mat_dtype : bool, optional
   If True, return arrays in same dtype as would be loaded into
   MATLAB (instead of the dtype with which they are saved).
squeeze_me : bool, optional
   Whether to squeeze unit matrix dimensions or not.
chars_as_strings : bool, optional
   Whether to convert char arrays to string arrays.
matlab_compatible : bool, optional
   Returns matrices as would be loaded by MATLAB (implies
   squeeze_me=False, chars_as_strings=False, mat_dtype=True,
   struct_as_record=True).
struct_as_record : bool, optional
   Whether to load MATLAB structs as NumPy record arrays, or as
   old-style NumPy arrays with dtype=object. Setting this flag to
   False replicates the behavior of SciPy version 0.7.x (returning
   numpy object arrays). The default setting is True, because it
   allows easier round-trip load and save of MATLAB files.

Returns
-------
variables : list of tuples
    A list of tuples, where each tuple holds the matrix name (a string),
    its shape (tuple of ints), and its data class (a string).
    Possible data classes are: int8, uint8, int16, uint16, int32, uint32,
    int64, uint64, single, double, cell, struct, object, char, sparse,
    function, opaque, logical, unknown.

Notes
-----
v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.

You will need an HDF5 python library to read matlab 7.3 format mat
files. Because SciPy does not supply one, we do not implement the
HDF5 / 7.3 interface here.

.. versionadded:: 0.12.0
*)


end

module Mmio : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module MMFile : sig
type tag = [`MMFile]
type t = [`MMFile | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?kwargs:(string * Py.Object.t) list -> unit -> t
(**
None
*)

val info : source:[`File_like of Py.Object.t | `S of string] -> [> tag] Obj.t -> (int * int * int * string * string * string)
(**
Return size, storage parameters from Matrix Market file-like 'source'.

Parameters
----------
source : str or file-like
    Matrix Market filename (extension .mtx) or open file-like object

Returns
-------
rows : int
    Number of matrix rows.
cols : int
    Number of matrix columns.
entries : int
    Number of non-zero entries of a sparse matrix
    or rows*cols for a dense matrix.
format : str
    Either 'coordinate' or 'array'.
field : str
    Either 'real', 'complex', 'pattern', or 'integer'.
symmetry : str
    Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
*)

val read : source:[`File_like of Py.Object.t | `S of string] -> [> tag] Obj.t -> Py.Object.t
(**
Reads the contents of a Matrix Market file-like 'source' into a matrix.

Parameters
----------
source : str or file-like
    Matrix Market filename (extensions .mtx, .mtz.gz)
    or open file object.

Returns
-------
a : ndarray or coo_matrix
    Dense or sparse matrix depending on the matrix format in the
    Matrix Market file.
*)

val write : ?comment:string -> ?field:string -> ?precision:int -> ?symmetry:string -> target:[`File_like of Py.Object.t | `S of string] -> a:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Writes sparse or dense array `a` to Matrix Market file-like `target`.

Parameters
----------
target : str or file-like
    Matrix Market filename (extension .mtx) or open file-like object.
a : array like
    Sparse or dense 2-D array.
comment : str, optional
    Comments to be prepended to the Matrix Market file.
field : None or str, optional
    Either 'real', 'complex', 'pattern', or 'integer'.
precision : None or int, optional
    Number of digits to display for real or complex values.
symmetry : None or str, optional
    Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
    If symmetry is None the symmetry type of 'a' is determined by its
    values.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Coo_matrix : sig
type tag = [`Coo_matrix]
type t = [`ArrayLike | `Coo_matrix | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?shape:Py.Object.t -> ?dtype:Py.Object.t -> ?copy:Py.Object.t -> arg1:Py.Object.t -> unit -> t
(**
A sparse matrix in COOrdinate format.

Also known as the 'ijv' or 'triplet' format.

This can be instantiated in several ways:
    coo_matrix(D)
        with a dense matrix D

    coo_matrix(S)
        with another sparse matrix S (equivalent to S.tocoo())

    coo_matrix((M, N), [dtype])
        to construct an empty matrix with shape (M, N)
        dtype is optional, defaulting to dtype='d'.

    coo_matrix((data, (i, j)), [shape=(M, N)])
        to construct from three arrays:
            1. data[:]   the entries of the matrix, in any order
            2. i[:]      the row indices of the matrix entries
            3. j[:]      the column indices of the matrix entries

        Where ``A[i[k], j[k]] = data[k]``.  When shape is not
        specified, it is inferred from the index arrays

Attributes
----------
dtype : dtype
    Data type of the matrix
shape : 2-tuple
    Shape of the matrix
ndim : int
    Number of dimensions (this is always 2)
nnz
    Number of stored values, including explicit zeros
data
    COO format data array of the matrix
row
    COO format row index array of the matrix
col
    COO format column index array of the matrix

Notes
-----

Sparse matrices can be used in arithmetic operations: they support
addition, subtraction, multiplication, division, and matrix power.

Advantages of the COO format
    - facilitates fast conversion among sparse formats
    - permits duplicate entries (see example)
    - very fast conversion to and from CSR/CSC formats

Disadvantages of the COO format
    - does not directly support:
        + arithmetic operations
        + slicing

Intended Usage
    - COO is a fast format for constructing sparse matrices
    - Once a matrix has been constructed, convert to CSR or
      CSC format for fast arithmetic and matrix vector operations
    - By default when converting to CSR or CSC format, duplicate (i,j)
      entries will be summed together.  This facilitates efficient
      construction of finite element matrices and the like. (see example)

Examples
--------

>>> # Constructing an empty matrix
>>> from scipy.sparse import coo_matrix
>>> coo_matrix((3, 4), dtype=np.int8).toarray()
array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]], dtype=int8)

>>> # Constructing a matrix using ijv format
>>> row  = np.array([0, 3, 1, 0])
>>> col  = np.array([0, 3, 1, 2])
>>> data = np.array([4, 5, 7, 9])
>>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
array([[4, 0, 9, 0],
       [0, 7, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 5]])

>>> # Constructing a matrix with duplicate indices
>>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
>>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
>>> data = np.array([1, 1, 1, 1, 1, 1, 1])
>>> coo = coo_matrix((data, (row, col)), shape=(4, 4))
>>> # Duplicate indices are maintained until implicitly or explicitly summed
>>> np.max(coo.data)
1
>>> coo.toarray()
array([[3, 0, 1, 0],
       [0, 2, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 1]])
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
None
*)

val arcsin : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arcsin.

See `numpy.arcsin` for more information.
*)

val arcsinh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arcsinh.

See `numpy.arcsinh` for more information.
*)

val arctan : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arctan.

See `numpy.arctan` for more information.
*)

val arctanh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise arctanh.

See `numpy.arctanh` for more information.
*)

val argmax : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return indices of maximum elements along an axis.

Implicit zero elements are also taken into account. If there are
several maximum values, the index of the first occurrence is returned.

Parameters
----------
axis : {-2, -1, 0, 1, None}, optional
    Axis along which the argmax is computed. If None (default), index
    of the maximum element in the flatten data is returned.
out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
ind : numpy.matrix or int
    Indices of maximum elements. If matrix, its size along `axis` is 1.
*)

val argmin : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return indices of minimum elements along an axis.

Implicit zero elements are also taken into account. If there are
several minimum values, the index of the first occurrence is returned.

Parameters
----------
axis : {-2, -1, 0, 1, None}, optional
    Axis along which the argmin is computed. If None (default), index
    of the minimum element in the flatten data is returned.
out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
 ind : numpy.matrix or int
    Indices of minimum elements. If matrix, its size along `axis` is 1.
*)

val asformat : ?copy:bool -> format:[`S of string | `None] -> [> tag] Obj.t -> Py.Object.t
(**
Return this matrix in the passed format.

Parameters
----------
format : {str, None}
    The desired matrix format ('csr', 'csc', 'lil', 'dok', 'array', ...)
    or None for no conversion.
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : This matrix in the passed format.
*)

val asfptype : [> tag] Obj.t -> Py.Object.t
(**
Upcast matrix to a floating point format (if necessary)
*)

val astype : ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
(**
Cast the matrix elements to a specified type.

Parameters
----------
dtype : string or numpy dtype
    Typecode or data-type to which to cast the data.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur.
    Defaults to 'unsafe' for backwards compatibility.
    'no' means the data types should not be cast at all.
    'equiv' means only byte-order changes are allowed.
    'safe' means only casts which can preserve values are allowed.
    'same_kind' means only safe casts or casts within a kind,
    like float64 to float32, are allowed.
    'unsafe' means any data conversions may be done.
copy : bool, optional
    If `copy` is `False`, the result might share some memory with this
    matrix. If `copy` is `True`, it is guaranteed that the result and
    this matrix do not share any memory.
*)

val ceil : [> tag] Obj.t -> Py.Object.t
(**
Element-wise ceil.

See `numpy.ceil` for more information.
*)

val conj : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise complex conjugation.

If the matrix is of non-complex data type and `copy` is False,
this method does nothing and the data is not copied.

Parameters
----------
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : The element-wise complex conjugate.
*)

val conjugate : ?copy:bool -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise complex conjugation.

If the matrix is of non-complex data type and `copy` is False,
this method does nothing and the data is not copied.

Parameters
----------
copy : bool, optional
    If True, the result is guaranteed to not share data with self.

Returns
-------
A : The element-wise complex conjugate.
*)

val copy : [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of this matrix.

No data/indices will be shared between the returned value and current
matrix.
*)

val count_nonzero : [> tag] Obj.t -> Py.Object.t
(**
Number of non-zero entries, equivalent to

np.count_nonzero(a.toarray())

Unlike getnnz() and the nnz property, which return the number of stored
entries (the length of the data attribute), this method counts the
actual number of non-zero entries in data.
*)

val deg2rad : [> tag] Obj.t -> Py.Object.t
(**
Element-wise deg2rad.

See `numpy.deg2rad` for more information.
*)

val diagonal : ?k:int -> [> tag] Obj.t -> Py.Object.t
(**
Returns the kth diagonal of the matrix.

Parameters
----------
k : int, optional
    Which diagonal to get, corresponding to elements a[i, i+k].
    Default: 0 (the main diagonal).

    .. versionadded:: 1.0

See also
--------
numpy.diagonal : Equivalent numpy function.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> A.diagonal()
array([1, 0, 5])
>>> A.diagonal(k=1)
array([2, 3])
*)

val dot : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Ordinary dot product

Examples
--------
>>> import numpy as np
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> v = np.array([1, 0, -1])
>>> A.dot(v)
array([ 1, -3, -1], dtype=int64)
*)

val eliminate_zeros : [> tag] Obj.t -> Py.Object.t
(**
Remove zero entries from the matrix

This is an *in place* operation
*)

val expm1 : [> tag] Obj.t -> Py.Object.t
(**
Element-wise expm1.

See `numpy.expm1` for more information.
*)

val floor : [> tag] Obj.t -> Py.Object.t
(**
Element-wise floor.

See `numpy.floor` for more information.
*)

val getH : [> tag] Obj.t -> Py.Object.t
(**
Return the Hermitian transpose of this matrix.

See Also
--------
numpy.matrix.getH : NumPy's implementation of `getH` for matrices
*)

val get_shape : [> tag] Obj.t -> Py.Object.t
(**
Get shape of a matrix.
*)

val getcol : j:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of column j of the matrix, as an (m x 1) sparse
matrix (column vector).
*)

val getformat : [> tag] Obj.t -> Py.Object.t
(**
Format of a matrix representation as a string.
*)

val getmaxprint : [> tag] Obj.t -> Py.Object.t
(**
Maximum number of elements to display when printed.
*)

val getnnz : ?axis:[`One | `Zero] -> [> tag] Obj.t -> Py.Object.t
(**
Number of stored values, including explicit zeros.

Parameters
----------
axis : None, 0, or 1
    Select between the number of values across the whole matrix, in
    each column, or in each row.

See also
--------
count_nonzero : Number of non-zero entries
*)

val getrow : i:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Returns a copy of row i of the matrix, as a (1 x n) sparse
matrix (row vector).
*)

val log1p : [> tag] Obj.t -> Py.Object.t
(**
Element-wise log1p.

See `numpy.log1p` for more information.
*)

val max : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the maximum of the matrix or maximum along an axis.
This takes all elements into account, not just the non-zero ones.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the maximum over all the matrix elements, returning
    a scalar (i.e., `axis` = `None`).

out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except
    for the default value, as this argument is not used.

Returns
-------
amax : coo_matrix or scalar
    Maximum of `a`. If `axis` is None, the result is a scalar value.
    If `axis` is given, the result is a sparse.coo_matrix of dimension
    ``a.ndim - 1``.

See Also
--------
min : The minimum value of a sparse matrix along a given axis.
numpy.matrix.max : NumPy's implementation of 'max' for matrices
*)

val maximum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise maximum between this and another matrix.
*)

val mean : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the arithmetic mean along the specified axis.

Returns the average of the matrix elements. The average is taken
over all elements in the matrix by default, otherwise over the
specified axis. `float64` intermediate and return values are used
for integer inputs.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the mean is computed. The default is to compute
    the mean of all elements in the matrix (i.e., `axis` = `None`).
dtype : data-type, optional
    Type to use in computing the mean. For integer inputs, the default
    is `float64`; for floating point inputs, it is the same as the
    input dtype.

    .. versionadded:: 0.18.0

out : np.matrix, optional
    Alternative output matrix in which to place the result. It must
    have the same shape as the expected output, but the type of the
    output values will be cast if necessary.

    .. versionadded:: 0.18.0

Returns
-------
m : np.matrix

See Also
--------
numpy.matrix.mean : NumPy's implementation of 'mean' for matrices
*)

val min : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the minimum of the matrix or maximum along an axis.
This takes all elements into account, not just the non-zero ones.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the minimum over all the matrix elements, returning
    a scalar (i.e., `axis` = `None`).

out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

Returns
-------
amin : coo_matrix or scalar
    Minimum of `a`. If `axis` is None, the result is a scalar value.
    If `axis` is given, the result is a sparse.coo_matrix of dimension
    ``a.ndim - 1``.

See Also
--------
max : The maximum value of a sparse matrix along a given axis.
numpy.matrix.min : NumPy's implementation of 'min' for matrices
*)

val minimum : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Element-wise minimum between this and another matrix.
*)

val multiply : other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Point-wise multiplication by another matrix
        
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
(**
nonzero indices

Returns a tuple of arrays (row,col) containing the indices
of the non-zero elements of the matrix.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1,2,0],[0,0,3],[4,0,5]])
>>> A.nonzero()
(array([0, 0, 1, 2, 2]), array([0, 1, 2, 0, 2]))
*)

val power : ?dtype:Py.Object.t -> n:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
This function performs element-wise power.

Parameters
----------
n : n is a scalar

dtype : If dtype is not specified, the current dtype will be preserved.
*)

val rad2deg : [> tag] Obj.t -> Py.Object.t
(**
Element-wise rad2deg.

See `numpy.rad2deg` for more information.
*)

val reshape : ?kwargs:(string * Py.Object.t) list -> Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Object|`Spmatrix] Np.Obj.t
(**
reshape(self, shape, order='C', copy=False)

Gives a new shape to a sparse matrix without changing its data.

Parameters
----------
shape : length-2 tuple of ints
    The new shape should be compatible with the original shape.
order : {'C', 'F'}, optional
    Read the elements using this index order. 'C' means to read and
    write the elements using C-like index order; e.g., read entire first
    row, then second row, etc. 'F' means to read and write the elements
    using Fortran-like index order; e.g., read entire first column, then
    second column, etc.
copy : bool, optional
    Indicates whether or not attributes of self should be copied
    whenever possible. The degree to which attributes are copied varies
    depending on the type of sparse matrix being used.

Returns
-------
reshaped_matrix : sparse matrix
    A sparse matrix with the given `shape`, not necessarily of the same
    format as the current object.

See Also
--------
numpy.matrix.reshape : NumPy's implementation of 'reshape' for
                       matrices
*)

val resize : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
Resize the matrix in-place to dimensions given by ``shape``

Any elements that lie within the new shape will remain at the same
indices, while non-zero elements lying outside the new shape are
removed.

Parameters
----------
shape : (int, int)
    number of rows and columns in the new matrix

Notes
-----
The semantics are not identical to `numpy.ndarray.resize` or
`numpy.resize`. Here, the same data will be maintained at each index
before and after reshape, if that index is within the new bounds. In
numpy, resizing maintains contiguity of the array, moving elements
around in the logical matrix but not within a flattened representation.

We give no guarantees about whether the underlying data attributes
(arrays, etc.) will be modified in place or replaced with new objects.
*)

val rint : [> tag] Obj.t -> Py.Object.t
(**
Element-wise rint.

See `numpy.rint` for more information.
*)

val set_shape : shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
See `reshape`.
*)

val setdiag : ?k:int -> values:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Set diagonal or off-diagonal elements of the array.

Parameters
----------
values : array_like
    New values of the diagonal elements.

    Values may have any length. If the diagonal is longer than values,
    then the remaining diagonal entries will not be set. If values if
    longer than the diagonal, then the remaining values are ignored.

    If a scalar value is given, all of the diagonal is set to it.

k : int, optional
    Which off-diagonal to set, corresponding to elements a[i,i+k].
    Default: 0 (the main diagonal).
*)

val sign : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sign.

See `numpy.sign` for more information.
*)

val sin : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sin.

See `numpy.sin` for more information.
*)

val sinh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sinh.

See `numpy.sinh` for more information.
*)

val sqrt : [> tag] Obj.t -> Py.Object.t
(**
Element-wise sqrt.

See `numpy.sqrt` for more information.
*)

val sum : ?axis:[`One | `Zero | `PyObject of Py.Object.t] -> ?dtype:Np.Dtype.t -> ?out:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Sum the matrix elements over a given axis.

Parameters
----------
axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the sum of all the matrix elements, returning a scalar
    (i.e., `axis` = `None`).
dtype : dtype, optional
    The type of the returned matrix and of the accumulator in which
    the elements are summed.  The dtype of `a` is used by default
    unless `a` has an integer dtype of less precision than the default
    platform integer.  In that case, if `a` is signed then the platform
    integer is used while if `a` is unsigned then an unsigned integer
    of the same precision as the platform integer is used.

    .. versionadded:: 0.18.0

out : np.matrix, optional
    Alternative output matrix in which to place the result. It must
    have the same shape as the expected output, but the type of the
    output values will be cast if necessary.

    .. versionadded:: 0.18.0

Returns
-------
sum_along_axis : np.matrix
    A matrix with the same shape as `self`, with the specified
    axis removed.

See Also
--------
numpy.matrix.sum : NumPy's implementation of 'sum' for matrices
*)

val sum_duplicates : [> tag] Obj.t -> Py.Object.t
(**
Eliminate duplicate matrix entries by adding them together

This is an *in place* operation
*)

val tan : [> tag] Obj.t -> Py.Object.t
(**
Element-wise tan.

See `numpy.tan` for more information.
*)

val tanh : [> tag] Obj.t -> Py.Object.t
(**
Element-wise tanh.

See `numpy.tanh` for more information.
*)

val toarray : ?order:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
See the docstring for `spmatrix.toarray`.
*)

val tobsr : ?blocksize:Py.Object.t -> ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Block Sparse Row format.

With copy=False, the data/indices may be shared between this matrix and
the resultant bsr_matrix.

When blocksize=(R, C) is provided, it will be used for construction of
the bsr_matrix.
*)

val tocoo : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to COOrdinate format.

With copy=False, the data/indices may be shared between this matrix and
the resultant coo_matrix.
*)

val tocsc : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Column format

Duplicate entries will be summed together.

Examples
--------
>>> from numpy import array
>>> from scipy.sparse import coo_matrix
>>> row  = array([0, 0, 1, 3, 1, 0, 0])
>>> col  = array([0, 2, 1, 3, 1, 0, 0])
>>> data = array([1, 1, 1, 1, 1, 1, 1])
>>> A = coo_matrix((data, (row, col)), shape=(4, 4)).tocsc()
>>> A.toarray()
array([[3, 0, 1, 0],
       [0, 2, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 1]])
*)

val tocsr : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Compressed Sparse Row format

Duplicate entries will be summed together.

Examples
--------
>>> from numpy import array
>>> from scipy.sparse import coo_matrix
>>> row  = array([0, 0, 1, 3, 1, 0, 0])
>>> col  = array([0, 2, 1, 3, 1, 0, 0])
>>> data = array([1, 1, 1, 1, 1, 1, 1])
>>> A = coo_matrix((data, (row, col)), shape=(4, 4)).tocsr()
>>> A.toarray()
array([[3, 0, 1, 0],
       [0, 2, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 1]])
*)

val todense : ?order:[`C | `F] -> ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `T2_D of Py.Object.t] -> [> tag] Obj.t -> [>`ArrayLike] Np.Obj.t
(**
Return a dense matrix representation of this matrix.

Parameters
----------
order : {'C', 'F'}, optional
    Whether to store multi-dimensional data in C (row-major)
    or Fortran (column-major) order in memory. The default
    is 'None', indicating the NumPy default of C-ordered.
    Cannot be specified in conjunction with the `out`
    argument.

out : ndarray, 2-D, optional
    If specified, uses this array (or `numpy.matrix`) as the
    output buffer instead of allocating a new array to
    return. The provided array must have the same shape and
    dtype as the sparse matrix on which you are calling the
    method.

Returns
-------
arr : numpy.matrix, 2-D
    A NumPy matrix object with the same shape and containing
    the same data represented by the sparse matrix, with the
    requested memory order. If `out` was passed and was an
    array (rather than a `numpy.matrix`), it will be filled
    with the appropriate values and returned wrapped in a
    `numpy.matrix` object that shares the same memory.
*)

val todia : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to sparse DIAgonal format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dia_matrix.
*)

val todok : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to Dictionary Of Keys format.

With copy=False, the data/indices may be shared between this matrix and
the resultant dok_matrix.
*)

val tolil : ?copy:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Convert this matrix to List of Lists format.

With copy=False, the data/indices may be shared between this matrix and
the resultant lil_matrix.
*)

val transpose : ?axes:Py.Object.t -> ?copy:bool -> [> tag] Obj.t -> Py.Object.t
(**
Reverses the dimensions of the sparse matrix.

Parameters
----------
axes : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except
    for the default value.
copy : bool, optional
    Indicates whether or not attributes of `self` should be
    copied whenever possible. The degree to which attributes
    are copied varies depending on the type of sparse matrix
    being used.

Returns
-------
p : `self` with the dimensions reversed.

See Also
--------
numpy.matrix.transpose : NumPy's implementation of 'transpose'
                         for matrices
*)

val trunc : [> tag] Obj.t -> Py.Object.t
(**
Element-wise trunc.

See `numpy.trunc` for more information.
*)


(** Attribute dtype: get value or raise Not_found if None.*)
val dtype : t -> Np.Dtype.t

(** Attribute dtype: get value as an option. *)
val dtype_opt : t -> (Np.Dtype.t) option


(** Attribute shape: get value or raise Not_found if None.*)
val shape : t -> Py.Object.t

(** Attribute shape: get value as an option. *)
val shape_opt : t -> (Py.Object.t) option


(** Attribute ndim: get value or raise Not_found if None.*)
val ndim : t -> int

(** Attribute ndim: get value as an option. *)
val ndim_opt : t -> (int) option


(** Attribute nnz: get value or raise Not_found if None.*)
val nnz : t -> Py.Object.t

(** Attribute nnz: get value as an option. *)
val nnz_opt : t -> (Py.Object.t) option


(** Attribute data: get value or raise Not_found if None.*)
val data : t -> Py.Object.t

(** Attribute data: get value as an option. *)
val data_opt : t -> (Py.Object.t) option


(** Attribute row: get value or raise Not_found if None.*)
val row : t -> Py.Object.t

(** Attribute row: get value as an option. *)
val row_opt : t -> (Py.Object.t) option


(** Attribute col: get value or raise Not_found if None.*)
val col : t -> Py.Object.t

(** Attribute col: get value as an option. *)
val col_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Ndarray : sig
type tag = [`Ndarray]
type t = [`ArrayLike | `Ndarray | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?dtype:Np.Dtype.t -> ?buffer:Py.Object.t -> ?offset:int -> ?strides:int list -> ?order:[`C | `F] -> shape:int list -> unit -> t
(**
ndarray(shape, dtype=float, buffer=None, offset=0,
        strides=None, order=None)

An array object represents a multidimensional, homogeneous array
of fixed-size items.  An associated data-type object describes the
format of each element in the array (its byte-order, how many bytes it
occupies in memory, whether it is an integer, a floating point number,
or something else, etc.)

Arrays should be constructed using `array`, `zeros` or `empty` (refer
to the See Also section below).  The parameters given here refer to
a low-level method (`ndarray(...)`) for instantiating an array.

For more information, refer to the `numpy` module and examine the
methods and attributes of an array.

Parameters
----------
(for the __new__ method; see Notes below)

shape : tuple of ints
    Shape of created array.
dtype : data-type, optional
    Any object that can be interpreted as a numpy data type.
buffer : object exposing buffer interface, optional
    Used to fill the array with data.
offset : int, optional
    Offset of array data in buffer.
strides : tuple of ints, optional
    Strides of data in memory.
order : {'C', 'F'}, optional
    Row-major (C-style) or column-major (Fortran-style) order.

Attributes
----------
T : ndarray
    Transpose of the array.
data : buffer
    The array's elements, in memory.
dtype : dtype object
    Describes the format of the elements in the array.
flags : dict
    Dictionary containing information related to memory use, e.g.,
    'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.
flat : numpy.flatiter object
    Flattened version of the array as an iterator.  The iterator
    allows assignments, e.g., ``x.flat = 3`` (See `ndarray.flat` for
    assignment examples; TODO).
imag : ndarray
    Imaginary part of the array.
real : ndarray
    Real part of the array.
size : int
    Number of elements in the array.
itemsize : int
    The memory use of each array element in bytes.
nbytes : int
    The total number of bytes required to store the array data,
    i.e., ``itemsize * size``.
ndim : int
    The array's number of dimensions.
shape : tuple of ints
    Shape of the array.
strides : tuple of ints
    The step-size required to move from one element to the next in
    memory. For example, a contiguous ``(3, 4)`` array of type
    ``int16`` in C-order has strides ``(8, 2)``.  This implies that
    to move from element to element in memory requires jumps of 2 bytes.
    To move from row-to-row, one needs to jump 8 bytes at a time
    (``2 * 4``).
ctypes : ctypes object
    Class containing properties of the array needed for interaction
    with ctypes.
base : ndarray
    If the array is a view into another array, that array is its `base`
    (unless that array is also a view).  The `base` array is where the
    array data is actually stored.

See Also
--------
array : Construct an array.
zeros : Create an array, each element of which is zero.
empty : Create an array, but leave its allocated memory unchanged (i.e.,
        it contains 'garbage').
dtype : Create a data-type.

Notes
-----
There are two modes of creating an array using ``__new__``:

1. If `buffer` is None, then only `shape`, `dtype`, and `order`
   are used.
2. If `buffer` is an object exposing the buffer interface, then
   all keywords are interpreted.

No ``__init__`` method is needed because the array is fully initialized
after the ``__new__`` method.

Examples
--------
These examples illustrate the low-level `ndarray` constructor.  Refer
to the `See Also` section above for easier ways of constructing an
ndarray.

First mode, `buffer` is None:

>>> np.ndarray(shape=(2,2), dtype=float, order='F')
array([[0.0e+000, 0.0e+000], # random
       [     nan, 2.5e-323]])

Second mode:

>>> np.ndarray((2,), buffer=np.array([1,2,3]),
...            offset=np.int_().itemsize,
...            dtype=int) # offset = 1*itemsize, i.e. skip first element
array([2, 3])
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)

val __setitem__ : key:Py.Object.t -> value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Set self[key] to value.
*)

val all : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.all(axis=None, out=None, keepdims=False)

Returns True if all elements evaluate to True.

Refer to `numpy.all` for full documentation.

See Also
--------
numpy.all : equivalent function
*)

val any : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.any(axis=None, out=None, keepdims=False)

Returns True if any of the elements of `a` evaluate to True.

Refer to `numpy.any` for full documentation.

See Also
--------
numpy.any : equivalent function
*)

val argmax : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argmax(axis=None, out=None)

Return indices of the maximum values along the given axis.

Refer to `numpy.argmax` for full documentation.

See Also
--------
numpy.argmax : equivalent function
*)

val argmin : ?axis:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argmin(axis=None, out=None)

Return indices of the minimum values along the given axis of `a`.

Refer to `numpy.argmin` for detailed documentation.

See Also
--------
numpy.argmin : equivalent function
*)

val argpartition : ?axis:Py.Object.t -> ?kind:Py.Object.t -> ?order:Py.Object.t -> kth:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argpartition(kth, axis=-1, kind='introselect', order=None)

Returns the indices that would partition this array.

Refer to `numpy.argpartition` for full documentation.

.. versionadded:: 1.8.0

See Also
--------
numpy.argpartition : equivalent function
*)

val argsort : ?axis:Py.Object.t -> ?kind:Py.Object.t -> ?order:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.argsort(axis=-1, kind=None, order=None)

Returns the indices that would sort this array.

Refer to `numpy.argsort` for full documentation.

See Also
--------
numpy.argsort : equivalent function
*)

val astype : ?order:[`C | `F | `A | `K] -> ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> ?subok:Py.Object.t -> ?copy:bool -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)

Copy of the array, cast to a specified type.

Parameters
----------
dtype : str or dtype
    Typecode or data-type to which the array is cast.
order : {'C', 'F', 'A', 'K'}, optional
    Controls the memory layout order of the result.
    'C' means C order, 'F' means Fortran order, 'A'
    means 'F' order if all the arrays are Fortran contiguous,
    'C' order otherwise, and 'K' means as close to the
    order the array elements appear in memory as possible.
    Default is 'K'.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur. Defaults to 'unsafe'
    for backwards compatibility.

      * 'no' means the data types should not be cast at all.
      * 'equiv' means only byte-order changes are allowed.
      * 'safe' means only casts which can preserve values are allowed.
      * 'same_kind' means only safe casts or casts within a kind,
        like float64 to float32, are allowed.
      * 'unsafe' means any data conversions may be done.
subok : bool, optional
    If True, then sub-classes will be passed-through (default), otherwise
    the returned array will be forced to be a base-class array.
copy : bool, optional
    By default, astype always returns a newly allocated array. If this
    is set to false, and the `dtype`, `order`, and `subok`
    requirements are satisfied, the input array is returned instead
    of a copy.

Returns
-------
arr_t : ndarray
    Unless `copy` is False and the other conditions for returning the input
    array are satisfied (see description for `copy` input parameter), `arr_t`
    is a new array of the same shape as the input array, with dtype, order
    given by `dtype`, `order`.

Notes
-----
.. versionchanged:: 1.17.0
   Casting between a simple data type and a structured one is possible only
   for 'unsafe' casting.  Casting to multiple fields is allowed, but
   casting from multiple fields is not.

.. versionchanged:: 1.9.0
   Casting from numeric to string types in 'safe' casting mode requires
   that the string dtype length is long enough to store the max
   integer/float value converted.

Raises
------
ComplexWarning
    When casting from complex to float or int. To avoid this,
    one should use ``a.real.astype(t)``.

Examples
--------
>>> x = np.array([1, 2, 2.5])
>>> x
array([1. ,  2. ,  2.5])

>>> x.astype(int)
array([1, 2, 2])
*)

val byteswap : ?inplace:bool -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.byteswap(inplace=False)

Swap the bytes of the array elements

Toggle between low-endian and big-endian data representation by
returning a byteswapped array, optionally swapped in-place.
Arrays of byte-strings are not swapped. The real and imaginary
parts of a complex number are swapped individually.

Parameters
----------
inplace : bool, optional
    If ``True``, swap bytes in-place, default is ``False``.

Returns
-------
out : ndarray
    The byteswapped array. If `inplace` is ``True``, this is
    a view to self.

Examples
--------
>>> A = np.array([1, 256, 8755], dtype=np.int16)
>>> list(map(hex, A))
['0x1', '0x100', '0x2233']
>>> A.byteswap(inplace=True)
array([  256,     1, 13090], dtype=int16)
>>> list(map(hex, A))
['0x100', '0x1', '0x3322']

Arrays of byte-strings are not swapped

>>> A = np.array([b'ceg', b'fac'])
>>> A.byteswap()
array([b'ceg', b'fac'], dtype='|S3')

``A.newbyteorder().byteswap()`` produces an array with the same values
  but different representation in memory

>>> A = np.array([1, 2, 3])
>>> A.view(np.uint8)
array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
       0, 0], dtype=uint8)
>>> A.newbyteorder().byteswap(inplace=True)
array([1, 2, 3])
>>> A.view(np.uint8)
array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
       0, 3], dtype=uint8)
*)

val choose : ?out:Py.Object.t -> ?mode:Py.Object.t -> choices:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.choose(choices, out=None, mode='raise')

Use an index array to construct a new array from a set of choices.

Refer to `numpy.choose` for full documentation.

See Also
--------
numpy.choose : equivalent function
*)

val clip : ?min:Py.Object.t -> ?max:Py.Object.t -> ?out:Py.Object.t -> ?kwargs:(string * Py.Object.t) list -> [> tag] Obj.t -> Py.Object.t
(**
a.clip(min=None, max=None, out=None, **kwargs)

Return an array whose values are limited to ``[min, max]``.
One of max or min must be given.

Refer to `numpy.clip` for full documentation.

See Also
--------
numpy.clip : equivalent function
*)

val compress : ?axis:Py.Object.t -> ?out:Py.Object.t -> condition:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.compress(condition, axis=None, out=None)

Return selected slices of this array along given axis.

Refer to `numpy.compress` for full documentation.

See Also
--------
numpy.compress : equivalent function
*)

val conj : [> tag] Obj.t -> Py.Object.t
(**
a.conj()

Complex-conjugate all elements.

Refer to `numpy.conjugate` for full documentation.

See Also
--------
numpy.conjugate : equivalent function
*)

val conjugate : [> tag] Obj.t -> Py.Object.t
(**
a.conjugate()

Return the complex conjugate, element-wise.

Refer to `numpy.conjugate` for full documentation.

See Also
--------
numpy.conjugate : equivalent function
*)

val copy : ?order:[`C | `F | `A | `K] -> [> tag] Obj.t -> Py.Object.t
(**
a.copy(order='C')

Return a copy of the array.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    Controls the memory layout of the copy. 'C' means C-order,
    'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
    'C' otherwise. 'K' means match the layout of `a` as closely
    as possible. (Note that this function and :func:`numpy.copy` are very
    similar, but have different default values for their order=
    arguments.)

See also
--------
numpy.copy
numpy.copyto

Examples
--------
>>> x = np.array([[1,2,3],[4,5,6]], order='F')

>>> y = x.copy()

>>> x.fill(0)

>>> x
array([[0, 0, 0],
       [0, 0, 0]])

>>> y
array([[1, 2, 3],
       [4, 5, 6]])

>>> y.flags['C_CONTIGUOUS']
True
*)

val cumprod : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.cumprod(axis=None, dtype=None, out=None)

Return the cumulative product of the elements along the given axis.

Refer to `numpy.cumprod` for full documentation.

See Also
--------
numpy.cumprod : equivalent function
*)

val cumsum : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.cumsum(axis=None, dtype=None, out=None)

Return the cumulative sum of the elements along the given axis.

Refer to `numpy.cumsum` for full documentation.

See Also
--------
numpy.cumsum : equivalent function
*)

val diagonal : ?offset:Py.Object.t -> ?axis1:Py.Object.t -> ?axis2:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.diagonal(offset=0, axis1=0, axis2=1)

Return specified diagonals. In NumPy 1.9 the returned array is a
read-only view instead of a copy as in previous NumPy versions.  In
a future version the read-only restriction will be removed.

Refer to :func:`numpy.diagonal` for full documentation.

See Also
--------
numpy.diagonal : equivalent function
*)

val dot : ?out:Py.Object.t -> b:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.dot(b, out=None)

Dot product of two arrays.

Refer to `numpy.dot` for full documentation.

See Also
--------
numpy.dot : equivalent function

Examples
--------
>>> a = np.eye(2)
>>> b = np.ones((2, 2)) * 2
>>> a.dot(b)
array([[2.,  2.],
       [2.,  2.]])

This array method can be conveniently chained:

>>> a.dot(b).dot(b)
array([[8.,  8.],
       [8.,  8.]])
*)

val dump : file:[`S of string | `Path of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.dump(file)

Dump a pickle of the array to the specified file.
The array can be read back with pickle.load or numpy.load.

Parameters
----------
file : str or Path
    A string naming the dump file.

    .. versionchanged:: 1.17.0
        `pathlib.Path` objects are now accepted.
*)

val dumps : [> tag] Obj.t -> Py.Object.t
(**
a.dumps()

Returns the pickle of the array as a string.
pickle.loads or numpy.loads will convert the string back to an array.

Parameters
----------
None
*)

val fill : value:[`F of float | `I of int | `Bool of bool | `S of string] -> [> tag] Obj.t -> Py.Object.t
(**
a.fill(value)

Fill the array with a scalar value.

Parameters
----------
value : scalar
    All elements of `a` will be assigned this value.

Examples
--------
>>> a = np.array([1, 2])
>>> a.fill(0)
>>> a
array([0, 0])
>>> a = np.empty(2)
>>> a.fill(1)
>>> a
array([1.,  1.])
*)

val flatten : ?order:[`C | `F | `A | `K] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.flatten(order='C')

Return a copy of the array collapsed into one dimension.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    'C' means to flatten in row-major (C-style) order.
    'F' means to flatten in column-major (Fortran-
    style) order. 'A' means to flatten in column-major
    order if `a` is Fortran *contiguous* in memory,
    row-major order otherwise. 'K' means to flatten
    `a` in the order the elements occur in memory.
    The default is 'C'.

Returns
-------
y : ndarray
    A copy of the input array, flattened to one dimension.

See Also
--------
ravel : Return a flattened array.
flat : A 1-D flat iterator over the array.

Examples
--------
>>> a = np.array([[1,2], [3,4]])
>>> a.flatten()
array([1, 2, 3, 4])
>>> a.flatten('F')
array([1, 3, 2, 4])
*)

val getfield : ?offset:int -> dtype:[`S of string | `Dtype of Np.Dtype.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.getfield(dtype, offset=0)

Returns a field of the given array as a certain type.

A field is a view of the array data with a given data-type. The values in
the view are determined by the given type and the offset into the current
array in bytes. The offset needs to be such that the view dtype fits in the
array dtype; for example an array of dtype complex128 has 16-byte elements.
If taking a view with a 32-bit integer (4 bytes), the offset needs to be
between 0 and 12 bytes.

Parameters
----------
dtype : str or dtype
    The data type of the view. The dtype size of the view can not be larger
    than that of the array itself.
offset : int
    Number of bytes to skip before beginning the element view.

Examples
--------
>>> x = np.diag([1.+1.j]*2)
>>> x[1, 1] = 2 + 4.j
>>> x
array([[1.+1.j,  0.+0.j],
       [0.+0.j,  2.+4.j]])
>>> x.getfield(np.float64)
array([[1.,  0.],
       [0.,  2.]])

By choosing an offset of 8 bytes we can select the complex part of the
array for our view:

>>> x.getfield(np.float64, offset=8)
array([[1.,  0.],
       [0.,  4.]])
*)

val item : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
a.item( *args)

Copy an element of an array to a standard Python scalar and return it.

Parameters
----------
\*args : Arguments (variable number and type)

    * none: in this case, the method only works for arrays
      with one element (`a.size == 1`), which element is
      copied into a standard Python scalar object and returned.

    * int_type: this argument is interpreted as a flat index into
      the array, specifying which element to copy and return.

    * tuple of int_types: functions as does a single int_type argument,
      except that the argument is interpreted as an nd-index into the
      array.

Returns
-------
z : Standard Python scalar object
    A copy of the specified element of the array as a suitable
    Python scalar

Notes
-----
When the data type of `a` is longdouble or clongdouble, item() returns
a scalar array object because there is no available Python scalar that
would not lose information. Void arrays return a buffer object for item(),
unless fields are defined, in which case a tuple is returned.

`item` is very similar to a[args], except, instead of an array scalar,
a standard Python scalar is returned. This can be useful for speeding up
access to elements of the array and doing arithmetic on elements of the
array using Python's optimized math.

Examples
--------
>>> np.random.seed(123)
>>> x = np.random.randint(9, size=(3, 3))
>>> x
array([[2, 2, 6],
       [1, 3, 6],
       [1, 0, 1]])
>>> x.item(3)
1
>>> x.item(7)
0
>>> x.item((0, 1))
2
>>> x.item((2, 2))
1
*)

val itemset : Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
a.itemset( *args)

Insert scalar into an array (scalar is cast to array's dtype, if possible)

There must be at least 1 argument, and define the last argument
as *item*.  Then, ``a.itemset( *args)`` is equivalent to but faster
than ``a[args] = item``.  The item should be a scalar value and `args`
must select a single item in the array `a`.

Parameters
----------
\*args : Arguments
    If one argument: a scalar, only used in case `a` is of size 1.
    If two arguments: the last argument is the value to be set
    and must be a scalar, the first argument specifies a single array
    element location. It is either an int or a tuple.

Notes
-----
Compared to indexing syntax, `itemset` provides some speed increase
for placing a scalar into a particular location in an `ndarray`,
if you must do this.  However, generally this is discouraged:
among other problems, it complicates the appearance of the code.
Also, when using `itemset` (and `item`) inside a loop, be sure
to assign the methods to a local variable to avoid the attribute
look-up at each loop iteration.

Examples
--------
>>> np.random.seed(123)
>>> x = np.random.randint(9, size=(3, 3))
>>> x
array([[2, 2, 6],
       [1, 3, 6],
       [1, 0, 1]])
>>> x.itemset(4, 0)
>>> x.itemset((2, 2), 9)
>>> x
array([[2, 2, 6],
       [1, 0, 6],
       [1, 0, 9]])
*)

val max : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> ?initial:Py.Object.t -> ?where:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

Return the maximum along a given axis.

Refer to `numpy.amax` for full documentation.

See Also
--------
numpy.amax : equivalent function
*)

val mean : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.mean(axis=None, dtype=None, out=None, keepdims=False)

Returns the average of the array elements along given axis.

Refer to `numpy.mean` for full documentation.

See Also
--------
numpy.mean : equivalent function
*)

val min : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> ?initial:Py.Object.t -> ?where:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

Return the minimum along a given axis.

Refer to `numpy.amin` for full documentation.

See Also
--------
numpy.amin : equivalent function
*)

val newbyteorder : ?new_order:string -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
arr.newbyteorder(new_order='S')

Return the array with the same data viewed with a different byte order.

Equivalent to::

    arr.view(arr.dtype.newbytorder(new_order))

Changes are also made in all fields and sub-arrays of the array data
type.



Parameters
----------
new_order : string, optional
    Byte order to force; a value from the byte order specifications
    below. `new_order` codes can be any of:

    * 'S' - swap dtype from current to opposite endian
    * {'<', 'L'} - little endian
    * {'>', 'B'} - big endian
    * {'=', 'N'} - native order
    * {'|', 'I'} - ignore (no change to byte order)

    The default value ('S') results in swapping the current
    byte order. The code does a case-insensitive check on the first
    letter of `new_order` for the alternatives above.  For example,
    any of 'B' or 'b' or 'biggish' are valid to specify big-endian.


Returns
-------
new_arr : array
    New array object with the dtype reflecting given change to the
    byte order.
*)

val nonzero : [> tag] Obj.t -> Py.Object.t
(**
a.nonzero()

Return the indices of the elements that are non-zero.

Refer to `numpy.nonzero` for full documentation.

See Also
--------
numpy.nonzero : equivalent function
*)

val partition : ?axis:int -> ?kind:[`Introselect] -> ?order:[`S of string | `StringList of string list] -> kth:[`I of int | `Is of int list] -> [> tag] Obj.t -> Py.Object.t
(**
a.partition(kth, axis=-1, kind='introselect', order=None)

Rearranges the elements in the array in such a way that the value of the
element in kth position is in the position it would be in a sorted array.
All elements smaller than the kth element are moved before this element and
all equal or greater are moved behind it. The ordering of the elements in
the two partitions is undefined.

.. versionadded:: 1.8.0

Parameters
----------
kth : int or sequence of ints
    Element index to partition by. The kth element value will be in its
    final sorted position and all smaller elements will be moved before it
    and all equal or greater elements behind it.
    The order of all elements in the partitions is undefined.
    If provided with a sequence of kth it will partition all elements
    indexed by kth of them into their sorted position at once.
axis : int, optional
    Axis along which to sort. Default is -1, which means sort along the
    last axis.
kind : {'introselect'}, optional
    Selection algorithm. Default is 'introselect'.
order : str or list of str, optional
    When `a` is an array with fields defined, this argument specifies
    which fields to compare first, second, etc. A single field can
    be specified as a string, and not all fields need to be specified,
    but unspecified fields will still be used, in the order in which
    they come up in the dtype, to break ties.

See Also
--------
numpy.partition : Return a parititioned copy of an array.
argpartition : Indirect partition.
sort : Full sort.

Notes
-----
See ``np.partition`` for notes on the different algorithms.

Examples
--------
>>> a = np.array([3, 4, 2, 1])
>>> a.partition(3)
>>> a
array([2, 1, 3, 4])

>>> a.partition((1, 3))
>>> a
array([1, 2, 3, 4])
*)

val prod : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> ?initial:Py.Object.t -> ?where:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.prod(axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True)

Return the product of the array elements over the given axis

Refer to `numpy.prod` for full documentation.

See Also
--------
numpy.prod : equivalent function
*)

val ptp : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.ptp(axis=None, out=None, keepdims=False)

Peak to peak (maximum - minimum) value along a given axis.

Refer to `numpy.ptp` for full documentation.

See Also
--------
numpy.ptp : equivalent function
*)

val put : ?mode:Py.Object.t -> indices:Py.Object.t -> values:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.put(indices, values, mode='raise')

Set ``a.flat[n] = values[n]`` for all `n` in indices.

Refer to `numpy.put` for full documentation.

See Also
--------
numpy.put : equivalent function
*)

val ravel : ?order:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.ravel([order])

Return a flattened array.

Refer to `numpy.ravel` for full documentation.

See Also
--------
numpy.ravel : equivalent function

ndarray.flat : a flat iterator on the array.
*)

val repeat : ?axis:Py.Object.t -> repeats:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.repeat(repeats, axis=None)

Repeat elements of an array.

Refer to `numpy.repeat` for full documentation.

See Also
--------
numpy.repeat : equivalent function
*)

val reshape : ?order:Py.Object.t -> shape:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.reshape(shape, order='C')

Returns an array containing the same data with a new shape.

Refer to `numpy.reshape` for full documentation.

See Also
--------
numpy.reshape : equivalent function

Notes
-----
Unlike the free function `numpy.reshape`, this method on `ndarray` allows
the elements of the shape parameter to be passed in as separate arguments.
For example, ``a.reshape(10, 11)`` is equivalent to
``a.reshape((10, 11))``.
*)

val resize : ?refcheck:bool -> new_shape:[`T_n_ints of Py.Object.t | `TupleOfInts of int list] -> [> tag] Obj.t -> Py.Object.t
(**
a.resize(new_shape, refcheck=True)

Change shape and size of array in-place.

Parameters
----------
new_shape : tuple of ints, or `n` ints
    Shape of resized array.
refcheck : bool, optional
    If False, reference count will not be checked. Default is True.

Returns
-------
None

Raises
------
ValueError
    If `a` does not own its own data or references or views to it exist,
    and the data memory must be changed.
    PyPy only: will always raise if the data memory must be changed, since
    there is no reliable way to determine if references or views to it
    exist.

SystemError
    If the `order` keyword argument is specified. This behaviour is a
    bug in NumPy.

See Also
--------
resize : Return a new array with the specified shape.

Notes
-----
This reallocates space for the data area if necessary.

Only contiguous arrays (data elements consecutive in memory) can be
resized.

The purpose of the reference count check is to make sure you
do not use this array as a buffer for another Python object and then
reallocate the memory. However, reference counts can increase in
other ways so if you are sure that you have not shared the memory
for this array with another Python object, then you may safely set
`refcheck` to False.

Examples
--------
Shrinking an array: array is flattened (in the order that the data are
stored in memory), resized, and reshaped:

>>> a = np.array([[0, 1], [2, 3]], order='C')
>>> a.resize((2, 1))
>>> a
array([[0],
       [1]])

>>> a = np.array([[0, 1], [2, 3]], order='F')
>>> a.resize((2, 1))
>>> a
array([[0],
       [2]])

Enlarging an array: as above, but missing entries are filled with zeros:

>>> b = np.array([[0, 1], [2, 3]])
>>> b.resize(2, 3) # new_shape parameter doesn't have to be a tuple
>>> b
array([[0, 1, 2],
       [3, 0, 0]])

Referencing an array prevents resizing...

>>> c = a
>>> a.resize((1, 1))
Traceback (most recent call last):
...
ValueError: cannot resize an array that references or is referenced ...

Unless `refcheck` is False:

>>> a.resize((1, 1), refcheck=False)
>>> a
array([[0]])
>>> c
array([[0]])
*)

val round : ?decimals:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.round(decimals=0, out=None)

Return `a` with each element rounded to the given number of decimals.

Refer to `numpy.around` for full documentation.

See Also
--------
numpy.around : equivalent function
*)

val searchsorted : ?side:Py.Object.t -> ?sorter:Py.Object.t -> v:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.searchsorted(v, side='left', sorter=None)

Find indices where elements of v should be inserted in a to maintain order.

For full documentation, see `numpy.searchsorted`

See Also
--------
numpy.searchsorted : equivalent function
*)

val setfield : ?offset:int -> val_:Py.Object.t -> dtype:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.setfield(val, dtype, offset=0)

Put a value into a specified place in a field defined by a data-type.

Place `val` into `a`'s field defined by `dtype` and beginning `offset`
bytes into the field.

Parameters
----------
val : object
    Value to be placed in field.
dtype : dtype object
    Data-type of the field in which to place `val`.
offset : int, optional
    The number of bytes into the field at which to place `val`.

Returns
-------
None

See Also
--------
getfield

Examples
--------
>>> x = np.eye(3)
>>> x.getfield(np.float64)
array([[1.,  0.,  0.],
       [0.,  1.,  0.],
       [0.,  0.,  1.]])
>>> x.setfield(3, np.int32)
>>> x.getfield(np.int32)
array([[3, 3, 3],
       [3, 3, 3],
       [3, 3, 3]], dtype=int32)
>>> x
array([[1.0e+000, 1.5e-323, 1.5e-323],
       [1.5e-323, 1.0e+000, 1.5e-323],
       [1.5e-323, 1.5e-323, 1.0e+000]])
>>> x.setfield(np.eye(3), np.int32)
>>> x
array([[1.,  0.,  0.],
       [0.,  1.,  0.],
       [0.,  0.,  1.]])
*)

val setflags : ?write:bool -> ?align:bool -> ?uic:bool -> [> tag] Obj.t -> Py.Object.t
(**
a.setflags(write=None, align=None, uic=None)

Set array flags WRITEABLE, ALIGNED, (WRITEBACKIFCOPY and UPDATEIFCOPY),
respectively.

These Boolean-valued flags affect how numpy interprets the memory
area used by `a` (see Notes below). The ALIGNED flag can only
be set to True if the data is actually aligned according to the type.
The WRITEBACKIFCOPY and (deprecated) UPDATEIFCOPY flags can never be set
to True. The flag WRITEABLE can only be set to True if the array owns its
own memory, or the ultimate owner of the memory exposes a writeable buffer
interface, or is a string. (The exception for string is made so that
unpickling can be done without copying memory.)

Parameters
----------
write : bool, optional
    Describes whether or not `a` can be written to.
align : bool, optional
    Describes whether or not `a` is aligned properly for its type.
uic : bool, optional
    Describes whether or not `a` is a copy of another 'base' array.

Notes
-----
Array flags provide information about how the memory area used
for the array is to be interpreted. There are 7 Boolean flags
in use, only four of which can be changed by the user:
WRITEBACKIFCOPY, UPDATEIFCOPY, WRITEABLE, and ALIGNED.

WRITEABLE (W) the data area can be written to;

ALIGNED (A) the data and strides are aligned appropriately for the hardware
(as determined by the compiler);

UPDATEIFCOPY (U) (deprecated), replaced by WRITEBACKIFCOPY;

WRITEBACKIFCOPY (X) this array is a copy of some other array (referenced
by .base). When the C-API function PyArray_ResolveWritebackIfCopy is
called, the base array will be updated with the contents of this array.

All flags can be accessed using the single (upper case) letter as well
as the full name.

Examples
--------
>>> y = np.array([[3, 1, 7],
...               [2, 0, 0],
...               [8, 5, 9]])
>>> y
array([[3, 1, 7],
       [2, 0, 0],
       [8, 5, 9]])
>>> y.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : True
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
>>> y.setflags(write=0, align=0)
>>> y.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : True
  WRITEABLE : False
  ALIGNED : False
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
>>> y.setflags(uic=1)
Traceback (most recent call last):
  File '<stdin>', line 1, in <module>
ValueError: cannot set WRITEBACKIFCOPY flag to True
*)

val sort : ?axis:int -> ?kind:[`Quicksort | `Heapsort | `Stable | `Mergesort] -> ?order:[`S of string | `StringList of string list] -> [> tag] Obj.t -> Py.Object.t
(**
a.sort(axis=-1, kind=None, order=None)

Sort an array in-place. Refer to `numpy.sort` for full documentation.

Parameters
----------
axis : int, optional
    Axis along which to sort. Default is -1, which means sort along the
    last axis.
kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
    Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
    and 'mergesort' use timsort under the covers and, in general, the
    actual implementation will vary with datatype. The 'mergesort' option
    is retained for backwards compatibility.

    .. versionchanged:: 1.15.0.
       The 'stable' option was added.

order : str or list of str, optional
    When `a` is an array with fields defined, this argument specifies
    which fields to compare first, second, etc.  A single field can
    be specified as a string, and not all fields need be specified,
    but unspecified fields will still be used, in the order in which
    they come up in the dtype, to break ties.

See Also
--------
numpy.sort : Return a sorted copy of an array.
numpy.argsort : Indirect sort.
numpy.lexsort : Indirect stable sort on multiple keys.
numpy.searchsorted : Find elements in sorted array.
numpy.partition: Partial sort.

Notes
-----
See `numpy.sort` for notes on the different sorting algorithms.

Examples
--------
>>> a = np.array([[1,4], [3,1]])
>>> a.sort(axis=1)
>>> a
array([[1, 4],
       [1, 3]])
>>> a.sort(axis=0)
>>> a
array([[1, 3],
       [1, 4]])

Use the `order` keyword to specify a field to use when sorting a
structured array:

>>> a = np.array([('a', 2), ('c', 1)], dtype=[('x', 'S1'), ('y', int)])
>>> a.sort(order='y')
>>> a
array([(b'c', 1), (b'a', 2)],
      dtype=[('x', 'S1'), ('y', '<i8')])
*)

val squeeze : ?axis:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.squeeze(axis=None)

Remove single-dimensional entries from the shape of `a`.

Refer to `numpy.squeeze` for full documentation.

See Also
--------
numpy.squeeze : equivalent function
*)

val std : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?ddof:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.std(axis=None, dtype=None, out=None, ddof=0, keepdims=False)

Returns the standard deviation of the array elements along given axis.

Refer to `numpy.std` for full documentation.

See Also
--------
numpy.std : equivalent function
*)

val sum : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?keepdims:Py.Object.t -> ?initial:Py.Object.t -> ?where:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.sum(axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True)

Return the sum of the array elements over the given axis.

Refer to `numpy.sum` for full documentation.

See Also
--------
numpy.sum : equivalent function
*)

val swapaxes : axis1:Py.Object.t -> axis2:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.swapaxes(axis1, axis2)

Return a view of the array with `axis1` and `axis2` interchanged.

Refer to `numpy.swapaxes` for full documentation.

See Also
--------
numpy.swapaxes : equivalent function
*)

val take : ?axis:Py.Object.t -> ?out:Py.Object.t -> ?mode:Py.Object.t -> indices:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.take(indices, axis=None, out=None, mode='raise')

Return an array formed from the elements of `a` at the given indices.

Refer to `numpy.take` for full documentation.

See Also
--------
numpy.take : equivalent function
*)

val tobytes : ?order:[`C | `F | `None] -> [> tag] Obj.t -> Py.Object.t
(**
a.tobytes(order='C')

Construct Python bytes containing the raw data bytes in the array.

Constructs Python bytes showing a copy of the raw contents of
data memory. The bytes object can be produced in either 'C' or 'Fortran',
or 'Any' order (the default is 'C'-order). 'Any' order means C-order
unless the F_CONTIGUOUS flag in the array is set, in which case it
means 'Fortran' order.

.. versionadded:: 1.9.0

Parameters
----------
order : {'C', 'F', None}, optional
    Order of the data for multidimensional arrays:
    C, Fortran, or the same as for the original array.

Returns
-------
s : bytes
    Python bytes exhibiting a copy of `a`'s raw data.

Examples
--------
>>> x = np.array([[0, 1], [2, 3]], dtype='<u2')
>>> x.tobytes()
b'\x00\x00\x01\x00\x02\x00\x03\x00'
>>> x.tobytes('C') == x.tobytes()
True
>>> x.tobytes('F')
b'\x00\x00\x02\x00\x01\x00\x03\x00'
*)

val tofile : ?sep:string -> ?format:string -> fid:[`S of string | `PyObject of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
a.tofile(fid, sep='', format='%s')

Write array to a file as text or binary (default).

Data is always written in 'C' order, independent of the order of `a`.
The data produced by this method can be recovered using the function
fromfile().

Parameters
----------
fid : file or str or Path
    An open file object, or a string containing a filename.

    .. versionchanged:: 1.17.0
        `pathlib.Path` objects are now accepted.

sep : str
    Separator between array items for text output.
    If '' (empty), a binary file is written, equivalent to
    ``file.write(a.tobytes())``.
format : str
    Format string for text file output.
    Each entry in the array is formatted to text by first converting
    it to the closest Python type, and then using 'format' % item.

Notes
-----
This is a convenience function for quick storage of array data.
Information on endianness and precision is lost, so this method is not a
good choice for files intended to archive data or transport data between
machines with different endianness. Some of these problems can be overcome
by outputting the data as text files, at the expense of speed and file
size.

When fid is a file object, array contents are directly written to the
file, bypassing the file object's ``write`` method. As a result, tofile
cannot be used with files objects supporting compression (e.g., GzipFile)
or file-like objects that do not support ``fileno()`` (e.g., BytesIO).
*)

val tolist : [> tag] Obj.t -> Py.Object.t
(**
a.tolist()

Return the array as an ``a.ndim``-levels deep nested list of Python scalars.

Return a copy of the array data as a (nested) Python list.
Data items are converted to the nearest compatible builtin Python type, via
the `~numpy.ndarray.item` function.

If ``a.ndim`` is 0, then since the depth of the nested list is 0, it will
not be a list at all, but a simple Python scalar.

Parameters
----------
none

Returns
-------
y : object, or list of object, or list of list of object, or ...
    The possibly nested list of array elements.

Notes
-----
The array may be recreated via ``a = np.array(a.tolist())``, although this
may sometimes lose precision.

Examples
--------
For a 1D array, ``a.tolist()`` is almost the same as ``list(a)``,
except that ``tolist`` changes numpy scalars to Python scalars:

>>> a = np.uint32([1, 2])
>>> a_list = list(a)
>>> a_list
[1, 2]
>>> type(a_list[0])
<class 'numpy.uint32'>
>>> a_tolist = a.tolist()
>>> a_tolist
[1, 2]
>>> type(a_tolist[0])
<class 'int'>

Additionally, for a 2D array, ``tolist`` applies recursively:

>>> a = np.array([[1, 2], [3, 4]])
>>> list(a)
[array([1, 2]), array([3, 4])]
>>> a.tolist()
[[1, 2], [3, 4]]

The base case for this recursion is a 0D array:

>>> a = np.array(1)
>>> list(a)
Traceback (most recent call last):
  ...
TypeError: iteration over a 0-d array
>>> a.tolist()
1
*)

val tostring : ?order:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.tostring(order='C')

A compatibility alias for `tobytes`, with exactly the same behavior.

Despite its name, it returns `bytes` not `str`\ s.

.. deprecated:: 1.19.0
*)

val trace : ?offset:Py.Object.t -> ?axis1:Py.Object.t -> ?axis2:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)

Return the sum along diagonals of the array.

Refer to `numpy.trace` for full documentation.

See Also
--------
numpy.trace : equivalent function
*)

val transpose : Py.Object.t list -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
a.transpose( *axes)

Returns a view of the array with axes transposed.

For a 1-D array this has no effect, as a transposed vector is simply the
same vector. To convert a 1-D array into a 2D column vector, an additional
dimension must be added. `np.atleast2d(a).T` achieves this, as does
`a[:, np.newaxis]`.
For a 2-D array, this is a standard matrix transpose.
For an n-D array, if axes are given, their order indicates how the
axes are permuted (see Examples). If axes are not provided and
``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

Parameters
----------
axes : None, tuple of ints, or `n` ints

 * None or no argument: reverses the order of the axes.

 * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
   `i`-th axis becomes `a.transpose()`'s `j`-th axis.

 * `n` ints: same as an n-tuple of the same ints (this form is
   intended simply as a 'convenience' alternative to the tuple form)

Returns
-------
out : ndarray
    View of `a`, with axes suitably permuted.

See Also
--------
ndarray.T : Array property returning the array transposed.
ndarray.reshape : Give a new shape to an array without changing its data.

Examples
--------
>>> a = np.array([[1, 2], [3, 4]])
>>> a
array([[1, 2],
       [3, 4]])
>>> a.transpose()
array([[1, 3],
       [2, 4]])
>>> a.transpose((1, 0))
array([[1, 3],
       [2, 4]])
>>> a.transpose(1, 0)
array([[1, 3],
       [2, 4]])
*)

val var : ?axis:Py.Object.t -> ?dtype:Py.Object.t -> ?out:Py.Object.t -> ?ddof:Py.Object.t -> ?keepdims:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.var(axis=None, dtype=None, out=None, ddof=0, keepdims=False)

Returns the variance of the array elements, along given axis.

Refer to `numpy.var` for full documentation.

See Also
--------
numpy.var : equivalent function
*)

val view : ?dtype:[`Ndarray_sub_class of Py.Object.t | `Dtype of Np.Dtype.t] -> ?type_:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
a.view([dtype][, type])

New view of array with the same data.

.. note::
    Passing None for ``dtype`` is different from omitting the parameter,
    since the former invokes ``dtype(None)`` which is an alias for
    ``dtype('float_')``.

Parameters
----------
dtype : data-type or ndarray sub-class, optional
    Data-type descriptor of the returned view, e.g., float32 or int16.
    Omitting it results in the view having the same data-type as `a`.
    This argument can also be specified as an ndarray sub-class, which
    then specifies the type of the returned object (this is equivalent to
    setting the ``type`` parameter).
type : Python type, optional
    Type of the returned view, e.g., ndarray or matrix.  Again, omission
    of the parameter results in type preservation.

Notes
-----
``a.view()`` is used two different ways:

``a.view(some_dtype)`` or ``a.view(dtype=some_dtype)`` constructs a view
of the array's memory with a different data-type.  This can cause a
reinterpretation of the bytes of memory.

``a.view(ndarray_subclass)`` or ``a.view(type=ndarray_subclass)`` just
returns an instance of `ndarray_subclass` that looks at the same array
(same shape, dtype, etc.)  This does not cause a reinterpretation of the
memory.

For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of
bytes per entry than the previous dtype (for example, converting a
regular array to a structured array), then the behavior of the view
cannot be predicted just from the superficial appearance of ``a`` (shown
by ``print(a)``). It also depends on exactly how ``a`` is stored in
memory. Therefore if ``a`` is C-ordered versus fortran-ordered, versus
defined as a slice or transpose, etc., the view may give different
results.


Examples
--------
>>> x = np.array([(1, 2)], dtype=[('a', np.int8), ('b', np.int8)])

Viewing array data using a different type and dtype:

>>> y = x.view(dtype=np.int16, type=np.matrix)
>>> y
matrix([[513]], dtype=int16)
>>> print(type(y))
<class 'numpy.matrix'>

Creating a view on a structured array so it can be used in calculations

>>> x = np.array([(1, 2),(3,4)], dtype=[('a', np.int8), ('b', np.int8)])
>>> xv = x.view(dtype=np.int8).reshape(-1,2)
>>> xv
array([[1, 2],
       [3, 4]], dtype=int8)
>>> xv.mean(0)
array([2.,  3.])

Making changes to the view changes the underlying array

>>> xv[0,1] = 20
>>> x
array([(1, 20), (3,  4)], dtype=[('a', 'i1'), ('b', 'i1')])

Using a view to convert an array to a recarray:

>>> z = x.view(np.recarray)
>>> z.a
array([1, 3], dtype=int8)

Views share data:

>>> x[0] = (9, 10)
>>> z[0]
(9, 10)

Views that change the dtype size (bytes per entry) should normally be
avoided on arrays defined by slices, transposes, fortran-ordering, etc.:

>>> x = np.array([[1,2,3],[4,5,6]], dtype=np.int16)
>>> y = x[:, 0:2]
>>> y
array([[1, 2],
       [4, 5]], dtype=int16)
>>> y.view(dtype=[('width', np.int16), ('length', np.int16)])
Traceback (most recent call last):
    ...
ValueError: To change to a dtype of a different size, the array must be C-contiguous
>>> z = y.copy()
>>> z.view(dtype=[('width', np.int16), ('length', np.int16)])
array([[(1, 2)],
       [(4, 5)]], dtype=[('width', '<i2'), ('length', '<i2')])
*)


(** Attribute T: get value or raise Not_found if None.*)
val t : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute T: get value as an option. *)
val t_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute data: get value or raise Not_found if None.*)
val data : t -> Py.Object.t

(** Attribute data: get value as an option. *)
val data_opt : t -> (Py.Object.t) option


(** Attribute dtype: get value or raise Not_found if None.*)
val dtype : t -> Py.Object.t

(** Attribute dtype: get value as an option. *)
val dtype_opt : t -> (Py.Object.t) option


(** Attribute flags: get value or raise Not_found if None.*)
val flags : t -> Py.Object.t

(** Attribute flags: get value as an option. *)
val flags_opt : t -> (Py.Object.t) option


(** Attribute flat: get value or raise Not_found if None.*)
val flat : t -> Py.Object.t

(** Attribute flat: get value as an option. *)
val flat_opt : t -> (Py.Object.t) option


(** Attribute imag: get value or raise Not_found if None.*)
val imag : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute imag: get value as an option. *)
val imag_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute real: get value or raise Not_found if None.*)
val real : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute real: get value as an option. *)
val real_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute size: get value or raise Not_found if None.*)
val size : t -> int

(** Attribute size: get value as an option. *)
val size_opt : t -> (int) option


(** Attribute itemsize: get value or raise Not_found if None.*)
val itemsize : t -> int

(** Attribute itemsize: get value as an option. *)
val itemsize_opt : t -> (int) option


(** Attribute nbytes: get value or raise Not_found if None.*)
val nbytes : t -> int

(** Attribute nbytes: get value as an option. *)
val nbytes_opt : t -> (int) option


(** Attribute ndim: get value or raise Not_found if None.*)
val ndim : t -> int

(** Attribute ndim: get value as an option. *)
val ndim_opt : t -> (int) option


(** Attribute shape: get value or raise Not_found if None.*)
val shape : t -> int array

(** Attribute shape: get value as an option. *)
val shape_opt : t -> (int array) option


(** Attribute strides: get value or raise Not_found if None.*)
val strides : t -> int array

(** Attribute strides: get value as an option. *)
val strides_opt : t -> (int array) option


(** Attribute ctypes: get value or raise Not_found if None.*)
val ctypes : t -> Py.Object.t

(** Attribute ctypes: get value as an option. *)
val ctypes_opt : t -> (Py.Object.t) option


(** Attribute base: get value or raise Not_found if None.*)
val base : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute base: get value as an option. *)
val base_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val asarray : ?dtype:Np.Dtype.t -> ?order:[`C | `F] -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Convert the input to an array.

Parameters
----------
a : array_like
    Input data, in any form that can be converted to an array.  This
    includes lists, lists of tuples, tuples, tuples of tuples, tuples
    of lists and ndarrays.
dtype : data-type, optional
    By default, the data-type is inferred from the input data.
order : {'C', 'F'}, optional
    Whether to use row-major (C-style) or
    column-major (Fortran-style) memory representation.
    Defaults to 'C'.

Returns
-------
out : ndarray
    Array interpretation of `a`.  No copy is performed if the input
    is already an ndarray with matching dtype and order.  If `a` is a
    subclass of ndarray, a base class ndarray is returned.

See Also
--------
asanyarray : Similar function which passes through subclasses.
ascontiguousarray : Convert input to a contiguous array.
asfarray : Convert input to a floating point ndarray.
asfortranarray : Convert input to an ndarray with column-major
                 memory order.
asarray_chkfinite : Similar function which checks input for NaNs and Infs.
fromiter : Create an array from an iterator.
fromfunction : Construct an array by executing a function on grid
               positions.

Examples
--------
Convert a list into an array:

>>> a = [1, 2]
>>> np.asarray(a)
array([1, 2])

Existing arrays are not copied:

>>> a = np.array([1, 2])
>>> np.asarray(a) is a
True

If `dtype` is set, array is copied only if dtype does not match:

>>> a = np.array([1, 2], dtype=np.float32)
>>> np.asarray(a, dtype=np.float32) is a
True
>>> np.asarray(a, dtype=np.float64) is a
False

Contrary to `asanyarray`, ndarray subclasses are not passed through:

>>> issubclass(np.recarray, np.ndarray)
True
>>> a = np.array([(1.0, 2), (3.0, 4)], dtype='f4,i4').view(np.recarray)
>>> np.asarray(a) is a
False
>>> np.asanyarray(a) is a
True
*)

val asbytes : Py.Object.t -> Py.Object.t
(**
None
*)

val asstr : Py.Object.t -> Py.Object.t
(**
None
*)

val can_cast : ?casting:[`No | `Equiv | `Safe | `Same_kind | `Unsafe] -> from_:[`Bool of bool | `F of float | `Dtype_specifier of Py.Object.t | `S of string | `Ndarray of [>`Ndarray] Np.Obj.t | `I of int | `Dtype of Np.Dtype.t] -> to_:[`Dtype_specifier of Py.Object.t | `Dtype of Np.Dtype.t] -> unit -> bool
(**
can_cast(from_, to, casting='safe')

Returns True if cast between data types can occur according to the
casting rule.  If from is a scalar or array scalar, also returns
True if the scalar value can be cast without overflow or truncation
to an integer.

Parameters
----------
from_ : dtype, dtype specifier, scalar, or array
    Data type, scalar, or array to cast from.
to : dtype or dtype specifier
    Data type to cast to.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur.

      * 'no' means the data types should not be cast at all.
      * 'equiv' means only byte-order changes are allowed.
      * 'safe' means only casts which can preserve values are allowed.
      * 'same_kind' means only safe casts or casts within a kind,
        like float64 to float32, are allowed.
      * 'unsafe' means any data conversions may be done.

Returns
-------
out : bool
    True if cast can occur according to the casting rule.

Notes
-----
.. versionchanged:: 1.17.0
   Casting between a simple data type and a structured one is possible only
   for 'unsafe' casting.  Casting to multiple fields is allowed, but
   casting from multiple fields is not.

.. versionchanged:: 1.9.0
   Casting from numeric to string types in 'safe' casting mode requires
   that the string dtype length is long enough to store the maximum
   integer/float value converted.

See also
--------
dtype, result_type

Examples
--------
Basic examples

>>> np.can_cast(np.int32, np.int64)
True
>>> np.can_cast(np.float64, complex)
True
>>> np.can_cast(complex, float)
False

>>> np.can_cast('i8', 'f8')
True
>>> np.can_cast('i8', 'f4')
False
>>> np.can_cast('i4', 'S4')
False

Casting scalars

>>> np.can_cast(100, 'i1')
True
>>> np.can_cast(150, 'i1')
False
>>> np.can_cast(150, 'u1')
True

>>> np.can_cast(3.5e100, np.float32)
False
>>> np.can_cast(1000.0, np.float32)
True

Array scalar checks the value, array does not

>>> np.can_cast(np.array(1000.0), np.float32)
True
>>> np.can_cast(np.array([1000.0]), np.float32)
False

Using the casting rules

>>> np.can_cast('i8', 'i8', 'no')
True
>>> np.can_cast('<i8', '>i8', 'no')
False

>>> np.can_cast('<i8', '>i8', 'equiv')
True
>>> np.can_cast('<i4', '>i8', 'equiv')
False

>>> np.can_cast('<i4', '>i8', 'safe')
True
>>> np.can_cast('<i8', '>i4', 'safe')
False

>>> np.can_cast('<i8', '>i4', 'same_kind')
True
>>> np.can_cast('<i8', '>u4', 'same_kind')
False

>>> np.can_cast('<i8', '>u4', 'unsafe')
True
*)

val concatenate : ?axis:int -> ?out:[>`Ndarray] Np.Obj.t -> a:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
concatenate((a1, a2, ...), axis=0, out=None)

Join a sequence of arrays along an existing axis.

Parameters
----------
a1, a2, ... : sequence of array_like
    The arrays must have the same shape, except in the dimension
    corresponding to `axis` (the first, by default).
axis : int, optional
    The axis along which the arrays will be joined.  If axis is None,
    arrays are flattened before use.  Default is 0.
out : ndarray, optional
    If provided, the destination to place the result. The shape must be
    correct, matching that of what concatenate would have returned if no
    out argument were specified.

Returns
-------
res : ndarray
    The concatenated array.

See Also
--------
ma.concatenate : Concatenate function that preserves input masks.
array_split : Split an array into multiple sub-arrays of equal or
              near-equal size.
split : Split array into a list of multiple sub-arrays of equal size.
hsplit : Split array into multiple sub-arrays horizontally (column wise).
vsplit : Split array into multiple sub-arrays vertically (row wise).
dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).
stack : Stack a sequence of arrays along a new axis.
block : Assemble arrays from blocks.
hstack : Stack arrays in sequence horizontally (column wise).
vstack : Stack arrays in sequence vertically (row wise).
dstack : Stack arrays in sequence depth wise (along third dimension).
column_stack : Stack 1-D arrays as columns into a 2-D array.

Notes
-----
When one or more of the arrays to be concatenated is a MaskedArray,
this function will return a MaskedArray object instead of an ndarray,
but the input masks are *not* preserved. In cases where a MaskedArray
is expected as input, use the ma.concatenate function from the masked
array module instead.

Examples
--------
>>> a = np.array([[1, 2], [3, 4]])
>>> b = np.array([[5, 6]])
>>> np.concatenate((a, b), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
>>> np.concatenate((a, b.T), axis=1)
array([[1, 2, 5],
       [3, 4, 6]])
>>> np.concatenate((a, b), axis=None)
array([1, 2, 3, 4, 5, 6])

This function will not preserve masking of MaskedArray inputs.

>>> a = np.ma.arange(3)
>>> a[1] = np.ma.masked
>>> b = np.arange(2, 5)
>>> a
masked_array(data=[0, --, 2],
             mask=[False,  True, False],
       fill_value=999999)
>>> b
array([2, 3, 4])
>>> np.concatenate([a, b])
masked_array(data=[0, 1, 2, 2, 3, 4],
             mask=False,
       fill_value=999999)
>>> np.ma.concatenate([a, b])
masked_array(data=[0, --, 2, 2, 3, 4],
             mask=[False,  True, False, False, False, False],
       fill_value=999999)
*)

val conj : ?out:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple_of_ndarray_and_None of Py.Object.t] -> ?where:[>`Ndarray] Np.Obj.t -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
conjugate(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Return the complex conjugate, element-wise.

The complex conjugate of a complex number is obtained by changing the
sign of its imaginary part.

Parameters
----------
x : array_like
    Input value.
out : ndarray, None, or tuple of ndarray and None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
where : array_like, optional
    This condition is broadcast over the input. At locations where the
    condition is True, the `out` array will be set to the ufunc result.
    Elsewhere, the `out` array will retain its original value.
    Note that if an uninitialized `out` array is created via the default
    ``out=None``, locations within it where the condition is False will
    remain uninitialized.
**kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

Returns
-------
y : ndarray
    The complex conjugate of `x`, with same dtype as `y`.
    This is a scalar if `x` is a scalar.

Notes
-----
`conj` is an alias for `conjugate`:

>>> np.conj is np.conjugate
True

Examples
--------
>>> np.conjugate(1+2j)
(1-2j)

>>> x = np.eye(2) + 1j * np.eye(2)
>>> np.conjugate(x)
array([[ 1.-1.j,  0.-0.j],
       [ 0.-0.j,  1.-1.j]])
*)

val imag : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return the imaginary part of the complex argument.

Parameters
----------
val : array_like
    Input array.

Returns
-------
out : ndarray or scalar
    The imaginary component of the complex argument. If `val` is real,
    the type of `val` is used for the output.  If `val` has complex
    elements, the returned type is float.

See Also
--------
real, angle, real_if_close

Examples
--------
>>> a = np.array([1+2j, 3+4j, 5+6j])
>>> a.imag
array([2.,  4.,  6.])
>>> a.imag = np.array([8, 10, 12])
>>> a
array([1. +8.j,  3.+10.j,  5.+12.j])
>>> np.imag(1 + 1j)
1.0
*)

val isspmatrix : Py.Object.t -> Py.Object.t
(**
Is x of a sparse matrix type?

Parameters
----------
x
    object to check for being a sparse matrix

Returns
-------
bool
    True if x is a sparse matrix, False otherwise

Notes
-----
issparse and isspmatrix are aliases for the same function.

Examples
--------
>>> from scipy.sparse import csr_matrix, isspmatrix
>>> isspmatrix(csr_matrix([[5]]))
True

>>> from scipy.sparse import isspmatrix
>>> isspmatrix(5)
False
*)

val mminfo : [`File_like of Py.Object.t | `S of string] -> (int * int * int * string * string * string)
(**
Return size and storage parameters from Matrix Market file-like 'source'.

Parameters
----------
source : str or file-like
    Matrix Market filename (extension .mtx) or open file-like object

Returns
-------
rows : int
    Number of matrix rows.
cols : int
    Number of matrix columns.
entries : int
    Number of non-zero entries of a sparse matrix
    or rows*cols for a dense matrix.
format : str
    Either 'coordinate' or 'array'.
field : str
    Either 'real', 'complex', 'pattern', or 'integer'.
symmetry : str
    Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
*)

val mmread : [`File_like of Py.Object.t | `S of string] -> Py.Object.t
(**
Reads the contents of a Matrix Market file-like 'source' into a matrix.

Parameters
----------
source : str or file-like
    Matrix Market filename (extensions .mtx, .mtz.gz)
    or open file-like object.

Returns
-------
a : ndarray or coo_matrix
    Dense or sparse matrix depending on the matrix format in the
    Matrix Market file.
*)

val mmwrite : ?comment:string -> ?field:string -> ?precision:int -> ?symmetry:string -> target:[`File_like of Py.Object.t | `S of string] -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Writes the sparse or dense array `a` to Matrix Market file-like `target`.

Parameters
----------
target : str or file-like
    Matrix Market filename (extension .mtx) or open file-like object.
a : array like
    Sparse or dense 2-D array.
comment : str, optional
    Comments to be prepended to the Matrix Market file.
field : None or str, optional
    Either 'real', 'complex', 'pattern', or 'integer'.
precision : None or int, optional
    Number of digits to display for real or complex values.
symmetry : None or str, optional
    Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
    If symmetry is None the symmetry type of 'a' is determined by its
    values.
*)

val ones : ?dtype:Np.Dtype.t -> ?order:[`C | `F] -> shape:[`I of int | `Is of int list] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return a new array of given shape and type, filled with ones.

Parameters
----------
shape : int or sequence of ints
    Shape of the new array, e.g., ``(2, 3)`` or ``2``.
dtype : data-type, optional
    The desired data-type for the array, e.g., `numpy.int8`.  Default is
    `numpy.float64`.
order : {'C', 'F'}, optional, default: C
    Whether to store multi-dimensional data in row-major
    (C-style) or column-major (Fortran-style) order in
    memory.

Returns
-------
out : ndarray
    Array of ones with the given shape, dtype, and order.

See Also
--------
ones_like : Return an array of ones with shape and type of input.
empty : Return a new uninitialized array.
zeros : Return a new array setting values to zero.
full : Return a new array of given shape filled with value.


Examples
--------
>>> np.ones(5)
array([1., 1., 1., 1., 1.])

>>> np.ones((5,), dtype=int)
array([1, 1, 1, 1, 1])

>>> np.ones((2, 1))
array([[1.],
       [1.]])

>>> s = (2,2)
>>> np.ones(s)
array([[1.,  1.],
       [1.,  1.]])
*)

val real : [>`Ndarray] Np.Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Return the real part of the complex argument.

Parameters
----------
val : array_like
    Input array.

Returns
-------
out : ndarray or scalar
    The real component of the complex argument. If `val` is real, the type
    of `val` is used for the output.  If `val` has complex elements, the
    returned type is float.

See Also
--------
real_if_close, imag, angle

Examples
--------
>>> a = np.array([1+2j, 3+4j, 5+6j])
>>> a.real
array([1.,  3.,  5.])
>>> a.real = 9
>>> a
array([9.+2.j,  9.+4.j,  9.+6.j])
>>> a.real = np.array([9, 8, 7])
>>> a
array([9.+2.j,  8.+4.j,  7.+6.j])
>>> np.real(1 + 1j)
1.0
*)

val zeros : ?dtype:Np.Dtype.t -> ?order:[`C | `F] -> shape:int list -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
zeros(shape, dtype=float, order='C')

Return a new array of given shape and type, filled with zeros.

Parameters
----------
shape : int or tuple of ints
    Shape of the new array, e.g., ``(2, 3)`` or ``2``.
dtype : data-type, optional
    The desired data-type for the array, e.g., `numpy.int8`.  Default is
    `numpy.float64`.
order : {'C', 'F'}, optional, default: 'C'
    Whether to store multi-dimensional data in row-major
    (C-style) or column-major (Fortran-style) order in
    memory.

Returns
-------
out : ndarray
    Array of zeros with the given shape, dtype, and order.

See Also
--------
zeros_like : Return an array of zeros with shape and type of input.
empty : Return a new uninitialized array.
ones : Return a new array setting values to one.
full : Return a new array of given shape filled with value.

Examples
--------
>>> np.zeros(5)
array([ 0.,  0.,  0.,  0.,  0.])

>>> np.zeros((5,), dtype=int)
array([0, 0, 0, 0, 0])

>>> np.zeros((2, 1))
array([[ 0.],
       [ 0.]])

>>> s = (2,2)
>>> np.zeros(s)
array([[ 0.,  0.],
       [ 0.,  0.]])

>>> np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')]) # custom dtype
array([(0, 0), (0, 0)],
      dtype=[('x', '<i4'), ('y', '<i4')])
*)


end

module Netcdf : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module OrderedDict : sig
type tag = [`OrderedDict]
type t = [`Object | `OrderedDict] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val __getitem__ : y:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
x.__getitem__(y) <==> x[y]
*)

val __iter__ : [> tag] Obj.t -> Py.Object.t
(**
Implement iter(self).
*)

val __setitem__ : key:Py.Object.t -> value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Set self[key] to value.
*)

val fromkeys : ?value:Py.Object.t -> iterable:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Create a new ordered dictionary with keys from iterable and values set to value.
*)

val get : ?default:Py.Object.t -> key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the value for key if key is in the dictionary, else default.
*)

val move_to_end : ?last:Py.Object.t -> key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Move an existing element to the end (or beginning if last is false).

Raise KeyError if the element does not exist.
*)

val pop : ?d:Py.Object.t -> k:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
od.pop(k[,d]) -> v, remove specified key and return the corresponding
value.  If key is not found, d is returned if given, otherwise KeyError
is raised.
*)

val popitem : ?last:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Remove and return a (key, value) pair from the dictionary.

Pairs are returned in LIFO order if last is true or FIFO order if false.
*)

val setdefault : ?default:Py.Object.t -> key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Insert key with a value of default if key is not in the dictionary.

Return the value for key if key is in the dictionary, else default.
*)

val update : ?e:Py.Object.t -> ?f:(string * Py.Object.t) list -> [> tag] Obj.t -> Py.Object.t
(**
D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
In either case, this is followed by: for k in F:  D[k] = F[k]
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Dtype : sig
type tag = [`Dtype]
type t = [`Dtype | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?align:bool -> ?copy:bool -> obj:Py.Object.t -> unit -> t
(**
dtype(obj, align=False, copy=False)

Create a data type object.

A numpy array is homogeneous, and contains elements described by a
dtype object. A dtype object can be constructed from different
combinations of fundamental numeric types.

Parameters
----------
obj
    Object to be converted to a data type object.
align : bool, optional
    Add padding to the fields to match what a C compiler would output
    for a similar C-struct. Can be ``True`` only if `obj` is a dictionary
    or a comma-separated string. If a struct dtype is being created,
    this also sets a sticky alignment flag ``isalignedstruct``.
copy : bool, optional
    Make a new copy of the data-type object. If ``False``, the result
    may just be a reference to a built-in data-type object.

See also
--------
result_type

Examples
--------
Using array-scalar type:

>>> np.dtype(np.int16)
dtype('int16')

Structured type, one field name 'f1', containing int16:

>>> np.dtype([('f1', np.int16)])
dtype([('f1', '<i2')])

Structured type, one field named 'f1', in itself containing a structured
type with one field:

>>> np.dtype([('f1', [('f1', np.int16)])])
dtype([('f1', [('f1', '<i2')])])

Structured type, two fields: the first field contains an unsigned int, the
second an int32:

>>> np.dtype([('f1', np.uint64), ('f2', np.int32)])
dtype([('f1', '<u8'), ('f2', '<i4')])

Using array-protocol type strings:

>>> np.dtype([('a','f8'),('b','S10')])
dtype([('a', '<f8'), ('b', 'S10')])

Using comma-separated field formats.  The shape is (2,3):

>>> np.dtype('i4, (2,3)f8')
dtype([('f0', '<i4'), ('f1', '<f8', (2, 3))])

Using tuples.  ``int`` is a fixed type, 3 the field's shape.  ``void``
is a flexible type, here of size 10:

>>> np.dtype([('hello',(np.int64,3)),('world',np.void,10)])
dtype([('hello', '<i8', (3,)), ('world', 'V10')])

Subdivide ``int16`` into 2 ``int8``'s, called x and y.  0 and 1 are
the offsets in bytes:

>>> np.dtype((np.int16, {'x':(np.int8,0), 'y':(np.int8,1)}))
dtype((numpy.int16, [('x', 'i1'), ('y', 'i1')]))

Using dictionaries.  Two fields named 'gender' and 'age':

>>> np.dtype({'names':['gender','age'], 'formats':['S1',np.uint8]})
dtype([('gender', 'S1'), ('age', 'u1')])

Offsets in bytes, here 0 and 25:

>>> np.dtype({'surname':('S25',0),'age':(np.uint8,25)})
dtype([('surname', 'S25'), ('age', 'u1')])
*)

val __getitem__ : key:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return self[key].
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val array : ?dtype:Np.Dtype.t -> ?copy:bool -> ?order:[`K | `A | `C | `F] -> ?subok:bool -> ?ndmin:int -> object_:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0)

Create an array.

Parameters
----------
object : array_like
    An array, any object exposing the array interface, an object whose
    __array__ method returns an array, or any (nested) sequence.
dtype : data-type, optional
    The desired data-type for the array.  If not given, then the type will
    be determined as the minimum type required to hold the objects in the
    sequence.
copy : bool, optional
    If true (default), then the object is copied.  Otherwise, a copy will
    only be made if __array__ returns a copy, if obj is a nested sequence,
    or if a copy is needed to satisfy any of the other requirements
    (`dtype`, `order`, etc.).
order : {'K', 'A', 'C', 'F'}, optional
    Specify the memory layout of the array. If object is not an array, the
    newly created array will be in C order (row major) unless 'F' is
    specified, in which case it will be in Fortran order (column major).
    If object is an array the following holds.

    ===== ========= ===================================================
    order  no copy                     copy=True
    ===== ========= ===================================================
    'K'   unchanged F & C order preserved, otherwise most similar order
    'A'   unchanged F order if input is F and not C, otherwise C order
    'C'   C order   C order
    'F'   F order   F order
    ===== ========= ===================================================

    When ``copy=False`` and a copy is made for other reasons, the result is
    the same as if ``copy=True``, with some exceptions for `A`, see the
    Notes section. The default order is 'K'.
subok : bool, optional
    If True, then sub-classes will be passed-through, otherwise
    the returned array will be forced to be a base-class array (default).
ndmin : int, optional
    Specifies the minimum number of dimensions that the resulting
    array should have.  Ones will be pre-pended to the shape as
    needed to meet this requirement.

Returns
-------
out : ndarray
    An array object satisfying the specified requirements.

See Also
--------
empty_like : Return an empty array with shape and type of input.
ones_like : Return an array of ones with shape and type of input.
zeros_like : Return an array of zeros with shape and type of input.
full_like : Return a new array with shape of input filled with value.
empty : Return a new uninitialized array.
ones : Return a new array setting values to one.
zeros : Return a new array setting values to zero.
full : Return a new array of given shape filled with value.


Notes
-----
When order is 'A' and `object` is an array in neither 'C' nor 'F' order,
and a copy is forced by a change in dtype, then the order of the result is
not necessarily 'C' as expected. This is likely a bug.

Examples
--------
>>> np.array([1, 2, 3])
array([1, 2, 3])

Upcasting:

>>> np.array([1, 2, 3.0])
array([ 1.,  2.,  3.])

More than one dimension:

>>> np.array([[1, 2], [3, 4]])
array([[1, 2],
       [3, 4]])

Minimum dimensions 2:

>>> np.array([1, 2, 3], ndmin=2)
array([[1, 2, 3]])

Type provided:

>>> np.array([1, 2, 3], dtype=complex)
array([ 1.+0.j,  2.+0.j,  3.+0.j])

Data-type consisting of more than one element:

>>> x = np.array([(1,2),(3,4)],dtype=[('a','<i4'),('b','<i4')])
>>> x['a']
array([1, 3])

Creating an array from sub-classes:

>>> np.array(np.mat('1 2; 3 4'))
array([[1, 2],
       [3, 4]])

>>> np.array(np.mat('1 2; 3 4'), subok=True)
matrix([[1, 2],
        [3, 4]])
*)

val asarray : ?dtype:Np.Dtype.t -> ?order:[`C | `F] -> a:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Convert the input to an array.

Parameters
----------
a : array_like
    Input data, in any form that can be converted to an array.  This
    includes lists, lists of tuples, tuples, tuples of tuples, tuples
    of lists and ndarrays.
dtype : data-type, optional
    By default, the data-type is inferred from the input data.
order : {'C', 'F'}, optional
    Whether to use row-major (C-style) or
    column-major (Fortran-style) memory representation.
    Defaults to 'C'.

Returns
-------
out : ndarray
    Array interpretation of `a`.  No copy is performed if the input
    is already an ndarray with matching dtype and order.  If `a` is a
    subclass of ndarray, a base class ndarray is returned.

See Also
--------
asanyarray : Similar function which passes through subclasses.
ascontiguousarray : Convert input to a contiguous array.
asfarray : Convert input to a floating point ndarray.
asfortranarray : Convert input to an ndarray with column-major
                 memory order.
asarray_chkfinite : Similar function which checks input for NaNs and Infs.
fromiter : Create an array from an iterator.
fromfunction : Construct an array by executing a function on grid
               positions.

Examples
--------
Convert a list into an array:

>>> a = [1, 2]
>>> np.asarray(a)
array([1, 2])

Existing arrays are not copied:

>>> a = np.array([1, 2])
>>> np.asarray(a) is a
True

If `dtype` is set, array is copied only if dtype does not match:

>>> a = np.array([1, 2], dtype=np.float32)
>>> np.asarray(a, dtype=np.float32) is a
True
>>> np.asarray(a, dtype=np.float64) is a
False

Contrary to `asanyarray`, ndarray subclasses are not passed through:

>>> issubclass(np.recarray, np.ndarray)
True
>>> a = np.array([(1.0, 2), (3.0, 4)], dtype='f4,i4').view(np.recarray)
>>> np.asarray(a) is a
False
>>> np.asanyarray(a) is a
True
*)

val asbytes : Py.Object.t -> Py.Object.t
(**
None
*)

val asstr : Py.Object.t -> Py.Object.t
(**
None
*)

val empty : ?dtype:Np.Dtype.t -> ?order:[`C | `F] -> shape:[`I of int | `Tuple_of_int of Py.Object.t] -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
empty(shape, dtype=float, order='C')

Return a new array of given shape and type, without initializing entries.

Parameters
----------
shape : int or tuple of int
    Shape of the empty array, e.g., ``(2, 3)`` or ``2``.
dtype : data-type, optional
    Desired output data-type for the array, e.g, `numpy.int8`. Default is
    `numpy.float64`.
order : {'C', 'F'}, optional, default: 'C'
    Whether to store multi-dimensional data in row-major
    (C-style) or column-major (Fortran-style) order in
    memory.

Returns
-------
out : ndarray
    Array of uninitialized (arbitrary) data of the given shape, dtype, and
    order.  Object arrays will be initialized to None.

See Also
--------
empty_like : Return an empty array with shape and type of input.
ones : Return a new array setting values to one.
zeros : Return a new array setting values to zero.
full : Return a new array of given shape filled with value.


Notes
-----
`empty`, unlike `zeros`, does not set the array values to zero,
and may therefore be marginally faster.  On the other hand, it requires
the user to manually set all the values in the array, and should be
used with caution.

Examples
--------
>>> np.empty([2, 2])
array([[ -9.74499359e+001,   6.69583040e-309],
       [  2.13182611e-314,   3.06959433e-309]])         #uninitialized

>>> np.empty([2, 2], dtype=int)
array([[-1073741821, -1067949133],
       [  496041986,    19249760]])                     #uninitialized
*)

val frombuffer : ?dtype:Np.Dtype.t -> ?count:int -> ?offset:int -> buffer:Py.Object.t -> unit -> Py.Object.t
(**
frombuffer(buffer, dtype=float, count=-1, offset=0)

Interpret a buffer as a 1-dimensional array.

Parameters
----------
buffer : buffer_like
    An object that exposes the buffer interface.
dtype : data-type, optional
    Data-type of the returned array; default: float.
count : int, optional
    Number of items to read. ``-1`` means all data in the buffer.
offset : int, optional
    Start reading the buffer from this offset (in bytes); default: 0.

Notes
-----
If the buffer has data that is not in machine byte-order, this should
be specified as part of the data-type, e.g.::

  >>> dt = np.dtype(int)
  >>> dt = dt.newbyteorder('>')
  >>> np.frombuffer(buf, dtype=dt) # doctest: +SKIP

The data of the resulting array will not be byteswapped, but will be
interpreted correctly.

Examples
--------
>>> s = b'hello world'
>>> np.frombuffer(s, dtype='S1', count=5, offset=6)
array([b'w', b'o', b'r', b'l', b'd'], dtype='|S1')

>>> np.frombuffer(b'\x01\x02', dtype=np.uint8)
array([1, 2], dtype=uint8)
>>> np.frombuffer(b'\x01\x02\x03\x04\x05', dtype=np.uint8, count=3)
array([1, 2, 3], dtype=uint8)
*)

val mul : a:Py.Object.t -> b:Py.Object.t -> unit -> Py.Object.t
(**
Same as a * b.
*)

val reduce : ?initial:Py.Object.t -> function_:Py.Object.t -> sequence:Py.Object.t -> unit -> Py.Object.t
(**
reduce(function, sequence[, initial]) -> value

Apply a function of two arguments cumulatively to the items of a sequence,
from left to right, so as to reduce the sequence to a single value.
For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
((((1+2)+3)+4)+5).  If initial is present, it is placed before the items
of the sequence in the calculation, and serves as a default when the
sequence is empty.
*)


end

val hb_read : Py.Object.t -> Py.Object.t
(**
Read HB-format file.

Parameters
----------
path_or_open_file : path-like or file-like
    If a file-like object, it is used as-is. Otherwise, it is opened
    before reading.

Returns
-------
data : scipy.sparse.csc_matrix instance
    The data read from the HB file as a sparse matrix.

Notes
-----
At the moment not the full Harwell-Boeing format is supported. Supported
features are:

    - assembled, non-symmetric, real matrices
    - integer for pointer/indices
    - exponential format for float values, and int format

Examples
--------
We can read and write a harwell-boeing format file:

>>> from scipy.io.harwell_boeing import hb_read, hb_write
>>> from scipy.sparse import csr_matrix, eye
>>> data = csr_matrix(eye(3))  # create a sparse matrix
>>> hb_write('data.hb', data)  # write a hb file
>>> print(hb_read('data.hb'))  # read a hb file
  (0, 0)    1.0
  (1, 1)    1.0
  (2, 2)    1.0
*)

val hb_write : ?hb_info:Py.Object.t -> path_or_open_file:Py.Object.t -> m:[>`Spmatrix] Np.Obj.t -> unit -> Py.Object.t
(**
Write HB-format file.

Parameters
----------
path_or_open_file : path-like or file-like
    If a file-like object, it is used as-is. Otherwise, it is opened
    before writing.
m : sparse-matrix
    the sparse matrix to write
hb_info : HBInfo
    contains the meta-data for write

Returns
-------
None

Notes
-----
At the moment not the full Harwell-Boeing format is supported. Supported
features are:

    - assembled, non-symmetric, real matrices
    - integer for pointer/indices
    - exponential format for float values, and int format

Examples
--------
We can read and write a harwell-boeing format file:

>>> from scipy.io.harwell_boeing import hb_read, hb_write
>>> from scipy.sparse import csr_matrix, eye
>>> data = csr_matrix(eye(3))  # create a sparse matrix
>>> hb_write('data.hb', data)  # write a hb file
>>> print(hb_read('data.hb'))  # read a hb file
  (0, 0)    1.0
  (1, 1)    1.0
  (2, 2)    1.0
*)

val loadmat : ?mdict:Py.Object.t -> ?appendmat:bool -> ?kwargs:(string * Py.Object.t) list -> file_name:string -> unit -> Py.Object.t
(**
Load MATLAB file.

Parameters
----------
file_name : str
   Name of the mat file (do not need .mat extension if
   appendmat==True). Can also pass open file-like object.
mdict : dict, optional
    Dictionary in which to insert matfile variables.
appendmat : bool, optional
   True to append the .mat extension to the end of the given
   filename, if not already present.
byte_order : str or None, optional
   None by default, implying byte order guessed from mat
   file. Otherwise can be one of ('native', '=', 'little', '<',
   'BIG', '>').
mat_dtype : bool, optional
   If True, return arrays in same dtype as would be loaded into
   MATLAB (instead of the dtype with which they are saved).
squeeze_me : bool, optional
   Whether to squeeze unit matrix dimensions or not.
chars_as_strings : bool, optional
   Whether to convert char arrays to string arrays.
matlab_compatible : bool, optional
   Returns matrices as would be loaded by MATLAB (implies
   squeeze_me=False, chars_as_strings=False, mat_dtype=True,
   struct_as_record=True).
struct_as_record : bool, optional
   Whether to load MATLAB structs as NumPy record arrays, or as
   old-style NumPy arrays with dtype=object. Setting this flag to
   False replicates the behavior of scipy version 0.7.x (returning
   NumPy object arrays). The default setting is True, because it
   allows easier round-trip load and save of MATLAB files.
verify_compressed_data_integrity : bool, optional
    Whether the length of compressed sequences in the MATLAB file
    should be checked, to ensure that they are not longer than we expect.
    It is advisable to enable this (the default) because overlong
    compressed sequences in MATLAB files generally indicate that the
    files have experienced some sort of corruption.
variable_names : None or sequence
    If None (the default) - read all variables in file. Otherwise,
    `variable_names` should be a sequence of strings, giving names of the
    MATLAB variables to read from the file. The reader will skip any
    variable with a name not in this sequence, possibly saving some read
    processing.
simplify_cells : False, optional
    If True, return a simplified dict structure (which is useful if the mat
    file contains cell arrays). Note that this only affects the structure
    of the result and not its contents (which is identical for both output
    structures). If True, this automatically sets `struct_as_record` to
    False and `squeeze_me` to True, which is required to simplify cells.

Returns
-------
mat_dict : dict
   dictionary with variable names as keys, and loaded matrices as
   values.

Notes
-----
v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.

You will need an HDF5 Python library to read MATLAB 7.3 format mat
files. Because SciPy does not supply one, we do not implement the
HDF5 / 7.3 interface here.

Examples
--------
>>> from os.path import dirname, join as pjoin
>>> import scipy.io as sio

Get the filename for an example .mat file from the tests/data directory.

>>> data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
>>> mat_fname = pjoin(data_dir, 'testdouble_7.4_GLNX86.mat')

Load the .mat file contents.

>>> mat_contents = sio.loadmat(mat_fname)

The result is a dictionary, one key/value pair for each variable:

>>> sorted(mat_contents.keys())
['__globals__', '__header__', '__version__', 'testdouble']
>>> mat_contents['testdouble']
array([[0.        , 0.78539816, 1.57079633, 2.35619449, 3.14159265,
        3.92699082, 4.71238898, 5.49778714, 6.28318531]])

By default SciPy reads MATLAB structs as structured NumPy arrays where the
dtype fields are of type `object` and the names correspond to the MATLAB
struct field names. This can be disabled by setting the optional argument
`struct_as_record=False`.

Get the filename for an example .mat file that contains a MATLAB struct
called `teststruct` and load the contents.

>>> matstruct_fname = pjoin(data_dir, 'teststruct_7.4_GLNX86.mat')
>>> matstruct_contents = sio.loadmat(matstruct_fname)
>>> teststruct = matstruct_contents['teststruct']
>>> teststruct.dtype
dtype([('stringfield', 'O'), ('doublefield', 'O'), ('complexfield', 'O')])

The size of the structured array is the size of the MATLAB struct, not the
number of elements in any particular field. The shape defaults to 2-D
unless the optional argument `squeeze_me=True`, in which case all length 1
dimensions are removed.

>>> teststruct.size
1
>>> teststruct.shape
(1, 1)

Get the 'stringfield' of the first element in the MATLAB struct.

>>> teststruct[0, 0]['stringfield']
array(['Rats live on no evil star.'],
  dtype='<U26')

Get the first element of the 'doublefield'.

>>> teststruct['doublefield'][0, 0]
array([[ 1.41421356,  2.71828183,  3.14159265]])

Load the MATLAB struct, squeezing out length 1 dimensions, and get the item
from the 'complexfield'.

>>> matstruct_squeezed = sio.loadmat(matstruct_fname, squeeze_me=True)
>>> matstruct_squeezed['teststruct'].shape
()
>>> matstruct_squeezed['teststruct']['complexfield'].shape
()
>>> matstruct_squeezed['teststruct']['complexfield'].item()
array([ 1.41421356+1.41421356j,  2.71828183+2.71828183j,
    3.14159265+3.14159265j])
*)

val mminfo : [`File_like of Py.Object.t | `S of string] -> (int * int * int * string * string * string)
(**
Return size and storage parameters from Matrix Market file-like 'source'.

Parameters
----------
source : str or file-like
    Matrix Market filename (extension .mtx) or open file-like object

Returns
-------
rows : int
    Number of matrix rows.
cols : int
    Number of matrix columns.
entries : int
    Number of non-zero entries of a sparse matrix
    or rows*cols for a dense matrix.
format : str
    Either 'coordinate' or 'array'.
field : str
    Either 'real', 'complex', 'pattern', or 'integer'.
symmetry : str
    Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
*)

val mmread : [`File_like of Py.Object.t | `S of string] -> Py.Object.t
(**
Reads the contents of a Matrix Market file-like 'source' into a matrix.

Parameters
----------
source : str or file-like
    Matrix Market filename (extensions .mtx, .mtz.gz)
    or open file-like object.

Returns
-------
a : ndarray or coo_matrix
    Dense or sparse matrix depending on the matrix format in the
    Matrix Market file.
*)

val mmwrite : ?comment:string -> ?field:string -> ?precision:int -> ?symmetry:string -> target:[`File_like of Py.Object.t | `S of string] -> a:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Writes the sparse or dense array `a` to Matrix Market file-like `target`.

Parameters
----------
target : str or file-like
    Matrix Market filename (extension .mtx) or open file-like object.
a : array like
    Sparse or dense 2-D array.
comment : str, optional
    Comments to be prepended to the Matrix Market file.
field : None or str, optional
    Either 'real', 'complex', 'pattern', or 'integer'.
precision : None or int, optional
    Number of digits to display for real or complex values.
symmetry : None or str, optional
    Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
    If symmetry is None the symmetry type of 'a' is determined by its
    values.
*)

val readsav : ?idict:Py.Object.t -> ?python_dict:bool -> ?uncompressed_file_name:string -> ?verbose:bool -> file_name:string -> unit -> Py.Object.t
(**
Read an IDL .sav file.

Parameters
----------
file_name : str
    Name of the IDL save file.
idict : dict, optional
    Dictionary in which to insert .sav file variables.
python_dict : bool, optional
    By default, the object return is not a Python dictionary, but a
    case-insensitive dictionary with item, attribute, and call access
    to variables. To get a standard Python dictionary, set this option
    to True.
uncompressed_file_name : str, optional
    This option only has an effect for .sav files written with the
    /compress option. If a file name is specified, compressed .sav
    files are uncompressed to this file. Otherwise, readsav will use
    the `tempfile` module to determine a temporary filename
    automatically, and will remove the temporary file upon successfully
    reading it in.
verbose : bool, optional
    Whether to print out information about the save file, including
    the records read, and available variables.

Returns
-------
idl_dict : AttrDict or dict
    If `python_dict` is set to False (default), this function returns a
    case-insensitive dictionary with item, attribute, and call access
    to variables. If `python_dict` is set to True, this function
    returns a Python dictionary with all variable names in lowercase.
    If `idict` was specified, then variables are written to the
    dictionary specified, and the updated dictionary is returned.

Examples
--------
>>> from os.path import dirname, join as pjoin
>>> import scipy.io as sio
>>> from scipy.io import readsav

Get the filename for an example .sav file from the tests/data directory.

>>> data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
>>> sav_fname = pjoin(data_dir, 'array_float32_1d.sav')

Load the .sav file contents.

>>> sav_data = readsav(sav_fname)

Get keys of the .sav file contents.

>>> print(sav_data.keys())
dict_keys(['array1d'])

Access a content with a key.

>>> print(sav_data['array1d'])
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0.]
*)

val savemat : ?appendmat:bool -> ?format:[`T4 | `T5] -> ?long_field_names:bool -> ?do_compression:bool -> ?oned_as:[`Row | `Column] -> file_name:[`S of string | `File_like_object of Py.Object.t] -> mdict:Py.Object.t -> unit -> Py.Object.t
(**
Save a dictionary of names and arrays into a MATLAB-style .mat file.

This saves the array objects in the given dictionary to a MATLAB-
style .mat file.

Parameters
----------
file_name : str or file-like object
    Name of the .mat file (.mat extension not needed if ``appendmat ==
    True``).
    Can also pass open file_like object.
mdict : dict
    Dictionary from which to save matfile variables.
appendmat : bool, optional
    True (the default) to append the .mat extension to the end of the
    given filename, if not already present.
format : {'5', '4'}, string, optional
    '5' (the default) for MATLAB 5 and up (to 7.2),
    '4' for MATLAB 4 .mat files.
long_field_names : bool, optional
    False (the default) - maximum field name length in a structure is
    31 characters which is the documented maximum length.
    True - maximum field name length in a structure is 63 characters
    which works for MATLAB 7.6+.
do_compression : bool, optional
    Whether or not to compress matrices on write. Default is False.
oned_as : {'row', 'column'}, optional
    If 'column', write 1-D NumPy arrays as column vectors.
    If 'row', write 1-D NumPy arrays as row vectors.

Examples
--------
>>> from scipy.io import savemat
>>> a = np.arange(20)
>>> mdic = {'a': a, 'label': 'experiment'}
>>> mdic
{'a': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19]),
'label': 'experiment'}
>>> savemat('matlab_matrix.mat', mdic)
*)

val whosmat : ?appendmat:bool -> ?kwargs:(string * Py.Object.t) list -> file_name:string -> unit -> Py.Object.t
(**
List variables inside a MATLAB file.

Parameters
----------
file_name : str
   Name of the mat file (do not need .mat extension if
   appendmat==True) Can also pass open file-like object.
appendmat : bool, optional
   True to append the .mat extension to the end of the given
   filename, if not already present.
byte_order : str or None, optional
   None by default, implying byte order guessed from mat
   file. Otherwise can be one of ('native', '=', 'little', '<',
   'BIG', '>').
mat_dtype : bool, optional
   If True, return arrays in same dtype as would be loaded into
   MATLAB (instead of the dtype with which they are saved).
squeeze_me : bool, optional
   Whether to squeeze unit matrix dimensions or not.
chars_as_strings : bool, optional
   Whether to convert char arrays to string arrays.
matlab_compatible : bool, optional
   Returns matrices as would be loaded by MATLAB (implies
   squeeze_me=False, chars_as_strings=False, mat_dtype=True,
   struct_as_record=True).
struct_as_record : bool, optional
   Whether to load MATLAB structs as NumPy record arrays, or as
   old-style NumPy arrays with dtype=object. Setting this flag to
   False replicates the behavior of SciPy version 0.7.x (returning
   numpy object arrays). The default setting is True, because it
   allows easier round-trip load and save of MATLAB files.

Returns
-------
variables : list of tuples
    A list of tuples, where each tuple holds the matrix name (a string),
    its shape (tuple of ints), and its data class (a string).
    Possible data classes are: int8, uint8, int16, uint16, int32, uint32,
    int64, uint64, single, double, cell, struct, object, char, sparse,
    function, opaque, logical, unknown.

Notes
-----
v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.

You will need an HDF5 python library to read matlab 7.3 format mat
files. Because SciPy does not supply one, we do not implement the
HDF5 / 7.3 interface here.

.. versionadded:: 0.12.0
*)

