(lang dune 2.4)
(name sklearn)
; version field is optional
(version 0.0.1)

(generate_opam_files true)

(source (github lehy/ocaml-sklearn))
(license BSD-3-Clause)
(authors "Ronan Le Hy")
(maintainers "Ronan Le Hy <ronan.lehy@gmail.com>")

(package
 (name sklearn)
 (synopsis "scikit-learn machine learning library")
 (description "These are bindings to Python's scikit-learn.")
 (depends
  pyml))
