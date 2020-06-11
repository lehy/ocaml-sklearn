(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module ConstantWarning : sig
type tag = [`ConstantWarning]
type t = [`BaseException | `ConstantWarning | `Object] Obj.t
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

module Codata : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val find : ?sub:string -> ?disp:bool -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t option
(**
Return list of physical_constant keys containing a given string.

Parameters
----------
sub : str, unicode
    Sub-string to search keys for.  By default, return all keys.
disp : bool
    If True, print the keys that are found, and return None.
    Otherwise, return the list of keys without printing anything.

Returns
-------
keys : list or None
    If `disp` is False, the list of keys is returned.
    Otherwise, None is returned.

Examples
--------
>>> from scipy.constants import find, physical_constants

Which keys in the ``physical_constants`` dictionary contain 'boltzmann'?

>>> find('boltzmann')
['Boltzmann constant',
 'Boltzmann constant in Hz/K',
 'Boltzmann constant in eV/K',
 'Boltzmann constant in inverse meter per kelvin',
 'Stefan-Boltzmann constant']

Get the constant called 'Boltzmann constant in Hz/K':

>>> physical_constants['Boltzmann constant in Hz/K']
(20836619120.0, 'Hz K^-1', 0.0)

Find constants with 'radius' in the key:

>>> find('radius')
['Bohr radius',
 'classical electron radius',
 'deuteron rms charge radius',
 'proton rms charge radius']
>>> physical_constants['classical electron radius']
(2.8179403262e-15, 'm', 1.3e-24)
*)

val parse_constants_2002to2014 : Py.Object.t -> Py.Object.t
(**
None
*)

val parse_constants_2018toXXXX : Py.Object.t -> Py.Object.t
(**
None
*)

val precision : [`S of string | `Python_string of Py.Object.t] -> float
(**
Relative precision in physical_constants indexed by key

Parameters
----------
key : Python string or unicode
    Key in dictionary `physical_constants`

Returns
-------
prec : float
    Relative precision in `physical_constants` corresponding to `key`

Examples
--------
>>> from scipy import constants
>>> constants.precision(u'proton mass')
5.1e-37
*)

val sqrt : Py.Object.t -> Py.Object.t
(**
Return the square root of x.
*)

val unit : [`S of string | `Python_string of Py.Object.t] -> Py.Object.t
(**
Unit in physical_constants indexed by key

Parameters
----------
key : Python string or unicode
    Key in dictionary `physical_constants`

Returns
-------
unit : Python string
    Unit in `physical_constants` corresponding to `key`

Examples
--------
>>> from scipy import constants
>>> constants.unit(u'proton mass')
'kg'
*)

val value : [`S of string | `Python_string of Py.Object.t] -> float
(**
Value in physical_constants indexed by key

Parameters
----------
key : Python string or unicode
    Key in dictionary `physical_constants`

Returns
-------
value : float
    Value in `physical_constants` corresponding to `key`

Examples
--------
>>> from scipy import constants
>>> constants.value(u'elementary charge')
1.602176634e-19
*)


end

module Constants : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val convert_temperature : val_:[>`Ndarray] Np.Obj.t -> old_scale:string -> new_scale:string -> unit -> Py.Object.t
(**
Convert from a temperature scale to another one among Celsius, Kelvin,
Fahrenheit and Rankine scales.

Parameters
----------
val : array_like
    Value(s) of the temperature(s) to be converted expressed in the
    original scale.

old_scale: str
    Specifies as a string the original scale from which the temperature
    value(s) will be converted. Supported scales are Celsius ('Celsius',
    'celsius', 'C' or 'c'), Kelvin ('Kelvin', 'kelvin', 'K', 'k'),
    Fahrenheit ('Fahrenheit', 'fahrenheit', 'F' or 'f') and Rankine
    ('Rankine', 'rankine', 'R', 'r').

new_scale: str
    Specifies as a string the new scale to which the temperature
    value(s) will be converted. Supported scales are Celsius ('Celsius',
    'celsius', 'C' or 'c'), Kelvin ('Kelvin', 'kelvin', 'K', 'k'),
    Fahrenheit ('Fahrenheit', 'fahrenheit', 'F' or 'f') and Rankine
    ('Rankine', 'rankine', 'R', 'r').

Returns
-------
res : float or array of floats
    Value(s) of the converted temperature(s) expressed in the new scale.

Notes
-----
.. versionadded:: 0.18.0

Examples
--------
>>> from scipy.constants import convert_temperature
>>> convert_temperature(np.array([-40, 40.0]), 'Celsius', 'Kelvin')
array([ 233.15,  313.15])
*)

val lambda2nu : [>`Ndarray] Np.Obj.t -> Py.Object.t
(**
Convert wavelength to optical frequency

Parameters
----------
lambda_ : array_like
    Wavelength(s) to be converted.

Returns
-------
nu : float or array of floats
    Equivalent optical frequency.

Notes
-----
Computes ``nu = c / lambda`` where c = 299792458.0, i.e., the
(vacuum) speed of light in meters/second.

Examples
--------
>>> from scipy.constants import lambda2nu, speed_of_light
>>> lambda2nu(np.array((1, speed_of_light)))
array([  2.99792458e+08,   1.00000000e+00])
*)

val nu2lambda : [>`Ndarray] Np.Obj.t -> Py.Object.t
(**
Convert optical frequency to wavelength.

Parameters
----------
nu : array_like
    Optical frequency to be converted.

Returns
-------
lambda : float or array of floats
    Equivalent wavelength(s).

Notes
-----
Computes ``lambda = c / nu`` where c = 299792458.0, i.e., the
(vacuum) speed of light in meters/second.

Examples
--------
>>> from scipy.constants import nu2lambda, speed_of_light
>>> nu2lambda(np.array((1, speed_of_light)))
array([  2.99792458e+08,   1.00000000e+00])
*)


end

val convert_temperature : val_:[>`Ndarray] Np.Obj.t -> old_scale:string -> new_scale:string -> unit -> Py.Object.t
(**
Convert from a temperature scale to another one among Celsius, Kelvin,
Fahrenheit and Rankine scales.

Parameters
----------
val : array_like
    Value(s) of the temperature(s) to be converted expressed in the
    original scale.

old_scale: str
    Specifies as a string the original scale from which the temperature
    value(s) will be converted. Supported scales are Celsius ('Celsius',
    'celsius', 'C' or 'c'), Kelvin ('Kelvin', 'kelvin', 'K', 'k'),
    Fahrenheit ('Fahrenheit', 'fahrenheit', 'F' or 'f') and Rankine
    ('Rankine', 'rankine', 'R', 'r').

new_scale: str
    Specifies as a string the new scale to which the temperature
    value(s) will be converted. Supported scales are Celsius ('Celsius',
    'celsius', 'C' or 'c'), Kelvin ('Kelvin', 'kelvin', 'K', 'k'),
    Fahrenheit ('Fahrenheit', 'fahrenheit', 'F' or 'f') and Rankine
    ('Rankine', 'rankine', 'R', 'r').

Returns
-------
res : float or array of floats
    Value(s) of the converted temperature(s) expressed in the new scale.

Notes
-----
.. versionadded:: 0.18.0

Examples
--------
>>> from scipy.constants import convert_temperature
>>> convert_temperature(np.array([-40, 40.0]), 'Celsius', 'Kelvin')
array([ 233.15,  313.15])
*)

val find : ?sub:string -> ?disp:bool -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t option
(**
Return list of physical_constant keys containing a given string.

Parameters
----------
sub : str, unicode
    Sub-string to search keys for.  By default, return all keys.
disp : bool
    If True, print the keys that are found, and return None.
    Otherwise, return the list of keys without printing anything.

Returns
-------
keys : list or None
    If `disp` is False, the list of keys is returned.
    Otherwise, None is returned.

Examples
--------
>>> from scipy.constants import find, physical_constants

Which keys in the ``physical_constants`` dictionary contain 'boltzmann'?

>>> find('boltzmann')
['Boltzmann constant',
 'Boltzmann constant in Hz/K',
 'Boltzmann constant in eV/K',
 'Boltzmann constant in inverse meter per kelvin',
 'Stefan-Boltzmann constant']

Get the constant called 'Boltzmann constant in Hz/K':

>>> physical_constants['Boltzmann constant in Hz/K']
(20836619120.0, 'Hz K^-1', 0.0)

Find constants with 'radius' in the key:

>>> find('radius')
['Bohr radius',
 'classical electron radius',
 'deuteron rms charge radius',
 'proton rms charge radius']
>>> physical_constants['classical electron radius']
(2.8179403262e-15, 'm', 1.3e-24)
*)

val lambda2nu : [>`Ndarray] Np.Obj.t -> Py.Object.t
(**
Convert wavelength to optical frequency

Parameters
----------
lambda_ : array_like
    Wavelength(s) to be converted.

Returns
-------
nu : float or array of floats
    Equivalent optical frequency.

Notes
-----
Computes ``nu = c / lambda`` where c = 299792458.0, i.e., the
(vacuum) speed of light in meters/second.

Examples
--------
>>> from scipy.constants import lambda2nu, speed_of_light
>>> lambda2nu(np.array((1, speed_of_light)))
array([  2.99792458e+08,   1.00000000e+00])
*)

val nu2lambda : [>`Ndarray] Np.Obj.t -> Py.Object.t
(**
Convert optical frequency to wavelength.

Parameters
----------
nu : array_like
    Optical frequency to be converted.

Returns
-------
lambda : float or array of floats
    Equivalent wavelength(s).

Notes
-----
Computes ``lambda = c / nu`` where c = 299792458.0, i.e., the
(vacuum) speed of light in meters/second.

Examples
--------
>>> from scipy.constants import nu2lambda, speed_of_light
>>> nu2lambda(np.array((1, speed_of_light)))
array([  2.99792458e+08,   1.00000000e+00])
*)

val precision : [`S of string | `Python_string of Py.Object.t] -> float
(**
Relative precision in physical_constants indexed by key

Parameters
----------
key : Python string or unicode
    Key in dictionary `physical_constants`

Returns
-------
prec : float
    Relative precision in `physical_constants` corresponding to `key`

Examples
--------
>>> from scipy import constants
>>> constants.precision(u'proton mass')
5.1e-37
*)

val unit : [`S of string | `Python_string of Py.Object.t] -> Py.Object.t
(**
Unit in physical_constants indexed by key

Parameters
----------
key : Python string or unicode
    Key in dictionary `physical_constants`

Returns
-------
unit : Python string
    Unit in `physical_constants` corresponding to `key`

Examples
--------
>>> from scipy import constants
>>> constants.unit(u'proton mass')
'kg'
*)

val value : [`S of string | `Python_string of Py.Object.t] -> float
(**
Value in physical_constants indexed by key

Parameters
----------
key : Python string or unicode
    Key in dictionary `physical_constants`

Returns
-------
value : float
    Value in `physical_constants` corresponding to `key`

Examples
--------
>>> from scipy import constants
>>> constants.value(u'elementary charge')
1.602176634e-19
*)

