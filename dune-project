(lang dune 2.4)
(name sklearn)
(version 0.22-0.1.1)

(generate_opam_files true)

(source (github lehy/ocaml-sklearn))
(license BSD-3-Clause)
(authors "Ronan Le Hy")
(maintainers "Ronan Le Hy <ronan.lehy@gmail.com>")

(package
 (name sklearn)
 (synopsis "Scikit-learn machine learning library for OCaml")
 (description "Scikit-learn machine learning library for OCaml
These are bindings to Python's scikit-learn machine learning library:
- Simple and efficient tools for predictive data analysis
- Accessible to everybody, and reusable in various contexts
- Built on NumPy, SciPy, and matplotlib
- Open source, commercially usable - BSD license
")
 (depends
  (ocaml (>= 4.07.1))
  (ppx_expect :with-test)
  (owl :with-test)
  (re :with-test)
  (pyml (>= 20200222))))
