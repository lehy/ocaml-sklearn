(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module FPUModeChangeWarning : sig
type tag = [`FPUModeChangeWarning]
type t = [`BaseException | `FPUModeChangeWarning | `Object] Obj.t
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

module LooseVersion : sig
type tag = [`LooseVersion]
type t = [`LooseVersion | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?vstring:Py.Object.t -> unit -> t
(**
Version numbering for anarchists and software realists.
Implements the standard interface for version number classes as
described above.  A version number consists of a series of numbers,
separated by either periods or strings of letters.  When comparing
version numbers, the numeric components will be compared
numerically, and the alphabetic components lexically.  The following
are all valid version numbers, in no particular order:

    1.5.1
    1.5.2b2
    161
    3.10a
    8.02
    3.4j
    1996.07.12
    3.2.pl0
    3.1.1.6
    2g6
    11g
    0.960923
    2.2beta29
    1.13++
    5.5.kw
    2.0b1pl0

In fact, there is no such thing as an invalid version number under
this scheme; the rules for comparison are simple and predictable,
but may not always give the results you want (for some definition
of 'want').
*)

val parse : vstring:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
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

val check_fpu_mode : Py.Object.t -> Py.Object.t
(**
Check FPU mode was not changed during the test.
*)

val pytest_configure : Py.Object.t -> Py.Object.t
(**
None
*)

val pytest_runtest_setup : Py.Object.t -> Py.Object.t
(**
None
*)

