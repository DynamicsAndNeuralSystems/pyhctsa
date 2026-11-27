{#- Class page for the API reference.
    The class docstring documents the constructor's parameters, so __init__ is not documented again.
    Only members the class defines itself are shown: not those it inherits (from int, tuple or a base
    class), and not private ones.
    The summary tables link to the full entries that autodoc writes below them, on the same page. -#}
{%- set own_methods = methods | reject("in", inherited_members) | reject("equalto", "__init__") | list -%}
{%- set own_attributes = attributes | reject("in", inherited_members) | list -%}
{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :member-order: bysource

   {% block methods %}
   {% if own_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in own_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if own_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in own_attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
