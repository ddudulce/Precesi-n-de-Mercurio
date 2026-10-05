import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.animation import FuncAnimation


# ============================================================
# 1. CONSTANTES Y DATOS DE MERCURIO
# ============================================================

AU = 149_597_870_700.0              # Unidad astronómica [m]
c = 299_792_458.0                   # Velocidad de la luz [m/s]
GM_sun = 1.32712440041279419e20     # Parámetro gravitacional solar [m^3/s^2]

a_AU = 0.38709927                   # Semieje mayor de Mercurio [AU]
e = 0.20563593                      # Excentricidad
T_days = 87.969                     # Período sideral [días]

a = a_AU * AU
p = a * (1 - e**2)

epsilon = 3 * GM_sun / (p * c**2)

RAD_TO_ARCSEC = 206264.80624709636
ORBITS_PER_CENTURY = 36525.0 / T_days


# ============================================================
# 2. ECUACIÓN ORBITAL RELATIVISTA
# ============================================================

# La ecuación obtenida analíticamente es
#
#       y'' + y = 1 + epsilon*y^2
#
# con
#
#       y = p/r
#
# donde la variable independiente es phi.


def orbital_system(phi, state):
    """
    Sistema equivalente de primer orden para

        y'' + y = 1 + epsilon*y^2.
    """

    y, dy_dphi = state

    d2y_dphi2 = 1 + epsilon * y**2 - y

    return [dy_dphi, d2y_dphi2]


def perihelion_event(phi, state):
    """
    En el perihelio r es mínimo y, como y = p/r,
    y alcanza un máximo.

    Por tanto, el perihelio ocurre cuando dy/dphi = 0
    pasando de valores positivos a negativos.
    """

    return state[1]


perihelion_event.direction = -1
perihelion_event.terminal = False


# ============================================================
# 3. INTEGRACIÓN NUMÉRICA
# ============================================================

# Comenzamos exactamente en el perihelio.
#
# Para la órbita kepleriana:
#
#       y = 1 + e*cos(phi)
#
# así que
#
#       y(0)  = 1 + e
#       y'(0) = 0.

initial_state = [1 + e, 0.0]

# No es necesario integrar cientos de miles de órbitas para
# medir la precesión por revolución. Unas decenas bastan.

N_INTEGRATION = 80

phi_max = 2 * np.pi * (N_INTEGRATION + 2)

solution = solve_ivp(
    orbital_system,
    (0, phi_max),
    initial_state,
    method="DOP853",
    rtol=1e-12,
    atol=1e-14,
    max_step=0.05,
    events=perihelion_event
)


# ============================================================
# 4. MEDICIÓN NUMÉRICA DE LA PRECESIÓN
# ============================================================

perihelia = solution.t_events[0]

# Separación angular entre perihelios consecutivos.

angular_periods = np.diff(perihelia)

# En el problema newtoniano el siguiente perihelio aparece
# exactamente después de 2*pi. El exceso angular es la
# precesión relativista.

delta_phi_each = angular_periods - 2 * np.pi

delta_phi_num = np.mean(delta_phi_each)

delta_arcsec_orbit = (
    delta_phi_num * RAD_TO_ARCSEC
)

delta_arcsec_century = (
    delta_arcsec_orbit * ORBITS_PER_CENTURY
)


# ============================================================
# 5. RESULTADO ANALÍTICO
# ============================================================

delta_phi_analytic = 2 * np.pi * epsilon

delta_analytic_arcsec_orbit = (
    delta_phi_analytic * RAD_TO_ARCSEC
)

delta_analytic_arcsec_century = (
    delta_analytic_arcsec_orbit * ORBITS_PER_CENTURY
)

relative_error = abs(
    delta_phi_num - delta_phi_analytic
) / abs(delta_phi_analytic)


# ============================================================
# 6. RESULTADOS EN TERMINAL
# ============================================================

print()
print("=" * 58)
print(" PRECESIÓN RELATIVISTA DEL PERIHELIO DE MERCURIO")
print("=" * 58)
print()

print("Parámetros orbitales")
print("--------------------")
print(f"a                       = {a:.8e} m")
print(f"e                       = {e:.8f}")
print(f"p                       = {p:.8e} m")
print(f"epsilon                 = {epsilon:.10e}")
print()

print("Resultado analítico")
print("-------------------")
print(
    f"Delta phi por órbita    = "
    f"{delta_phi_analytic:.10e} rad"
)
print(
    f"Delta phi por órbita    = "
    f"{delta_analytic_arcsec_orbit:.8f} arcsec"
)
print(
    f"Delta phi por siglo     = "
    f"{delta_analytic_arcsec_century:.6f} arcsec/siglo"
)
print()

print("Resultado numérico")
print("------------------")
print(
    f"Delta phi por órbita    = "
    f"{delta_phi_num:.10e} rad"
)
print(
    f"Delta phi por órbita    = "
    f"{delta_arcsec_orbit:.8f} arcsec"
)
print(
    f"Delta phi por siglo     = "
    f"{delta_arcsec_century:.6f} arcsec/siglo"
)
print()

print(
    f"Error relativo          = "
    f"{relative_error:.3e}"
)
print()


# ============================================================
# 7. GRÁFICA ESTÁTICA PARA EL INFORME
# ============================================================

# Esta es la única figura estática principal que conservaremos.
#
# Compara la precesión acumulada obtenida mediante:
#
#   1. la aproximación analítica;
#   2. la integración numérica.
#
# El intervalo llega aproximadamente hasta un siglo.

orbit_number = np.linspace(
    0,
    ORBITS_PER_CENTURY,
    500
)

precession_numerical = (
    orbit_number
    * delta_phi_num
    * RAD_TO_ARCSEC
)

precession_analytic = (
    orbit_number
    * delta_phi_analytic
    * RAD_TO_ARCSEC
)


fig_static, ax_static = plt.subplots(
    figsize=(8, 5)
)

ax_static.plot(
    orbit_number,
    precession_numerical,
    linewidth=2,
    label="Integración numérica"
)

ax_static.plot(
    orbit_number,
    precession_analytic,
    linestyle="--",
    linewidth=1.8,
    label="Aproximación analítica"
)

ax_static.set_xlabel(
    "Número de revoluciones"
)

ax_static.set_ylabel(
    "Precesión acumulada [arcsec]"
)

ax_static.set_title(
    "Precesión acumulada del perihelio de Mercurio"
)

ax_static.legend()

fig_static.tight_layout()

fig_static.savefig(
    "precesion_acumulada.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig_static)


# ============================================================
# 8. ÓRBITA BASE PARA LA SIMULACIÓN
# ============================================================

# Para visualizar la evolución a largo plazo utilizamos la
# elipse de Mercurio con sus parámetros orbitales reales.
#
# La integración anterior determina cuánto avanza el perihelio
# por revolución. Después de N revoluciones, la orientación es
#
#       omega_N = N * Delta_phi_num.
#
# De esta manera NO se amplifica artificialmente epsilon.

theta = np.linspace(
    0,
    2 * np.pi,
    1600
)

r_base = p / (
    1 + e * np.cos(theta)
)

x_base = (
    r_base
    * np.cos(theta)
    / AU
)

y_base = (
    r_base
    * np.sin(theta)
    / AU
)

r_perihelion = (
    p / (1 + e)
)

r_perihelion_AU = (
    r_perihelion / AU
)


# ============================================================
# 9. FUNCIONES DE LA SIMULACIÓN
# ============================================================

def rotate_orbit(angle):
    """
    Rota la elipse un ángulo 'angle' alrededor del Sol.
    """

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    x_rot = (
        x_base * cos_angle
        - y_base * sin_angle
    )

    y_rot = (
        x_base * sin_angle
        + y_base * cos_angle
    )

    return x_rot, y_rot


def accumulated_precession(revolutions):
    """
    Precesión acumulada después de N revoluciones.
    """

    return revolutions * delta_phi_num


def equivalent_years(revolutions):
    """
    Tiempo físico equivalente al número de revoluciones.
    """

    return (
        revolutions
        * T_days
        / 365.25
    )


# ============================================================
# 10. CONSTRUCCIÓN DE LA VENTANA
# ============================================================

fig = plt.figure(
    figsize=(13.2, 8.2)
)


# ------------------------------------------------------------
# Región principal: órbita
# ------------------------------------------------------------

ax = fig.add_axes(
    [0.065, 0.25, 0.62, 0.68]
)

ax.set_aspect("equal")

ax.set_xlabel(
    "x [AU]",
    fontsize=11
)

ax.set_ylabel(
    "y [AU]",
    fontsize=11
)

# Dejamos suficiente espacio para que la órbita pueda rotar
# sin tocar los bordes.

ax.set_xlim(
    -0.52,
    0.42
)

ax.set_ylim(
    -0.47,
    0.47
)

ax.set_title(
    "Precesión relativista del perihelio de Mercurio",
    fontsize=15,
    pad=12
)


# ------------------------------------------------------------
# Panel derecho para leyenda e información
# ------------------------------------------------------------

side_panel = fig.add_axes(
    [0.725, 0.30, 0.25, 0.60]
)

side_panel.axis("off")


# ============================================================
# 11. ELEMENTOS DE LA ÓRBITA
# ============================================================

# Órbita newtoniana: permanece fija.

newton_line, = ax.plot(
    x_base,
    y_base,
    linestyle="--",
    linewidth=1.4,
    alpha=0.65,
    label="Órbita newtoniana"
)


# Órbita relativista.

gr_line, = ax.plot(
    x_base,
    y_base,
    linewidth=2.3,
    label="Órbita relativista"
)


# Sol.

sun = ax.scatter(
    [0],
    [0],
    s=75,
    label="Sol",
    zorder=5
)


# Dirección inicial del perihelio.

initial_apsis_line, = ax.plot(
    [0, r_perihelion_AU],
    [0, 0],
    linestyle=":",
    linewidth=1.8,
    label="Perihelio inicial"
)


# Dirección actual del perihelio.

current_apsis_line, = ax.plot(
    [0, r_perihelion_AU],
    [0, 0],
    linewidth=2.2,
    label="Perihelio actual"
)


# Mercurio.

mercury, = ax.plot(
    [r_perihelion_AU],
    [0],
    marker="o",
    markersize=7,
    linestyle="None",
    label="Mercurio"
)


# ============================================================
# 12. LEYENDA
# ============================================================

legend_handles = [
    newton_line,
    gr_line,
    sun,
    initial_apsis_line,
    current_apsis_line,
    mercury
]

side_panel.legend(
    handles=legend_handles,
    loc="upper left",
    frameon=True,
    fontsize=10,
    title="Elementos",
    title_fontsize=11
)


# ============================================================
# 13. PANEL DE INFORMACIÓN
# ============================================================

side_panel.text(
    0,
    0.53,
    "Estado de la simulación",
    fontsize=12,
    fontweight="bold",
    va="top"
)

info_text = side_panel.text(
    0,
    0.47,
    "",
    fontsize=10.5,
    va="top",
    linespacing=1.65
)


# ============================================================
# 14. SLIDER
# ============================================================

slider_ax = fig.add_axes(
    [0.135, 0.17, 0.515, 0.035]
)

revolution_slider = Slider(
    ax=slider_ax,
    label="Revoluciones",
    valmin=0,
    valmax=1_000_000,
    valinit=ORBITS_PER_CENTURY,
    valfmt="%0.0f"
)

# Ocultamos el valor automático del slider porque la cantidad
# exacta ya aparece claramente en el panel derecho.

revolution_slider.valtext.set_visible(False)


# ============================================================
# 15. CHECKBOX NEWTON
# ============================================================

check_ax = fig.add_axes(
    [0.755, 0.305, 0.15, 0.065]
)

check_newton = CheckButtons(
    check_ax,
    ["Newton"],
    [True]
)

check_ax.set_title(
    "Referencia",
    fontsize=10,
    pad=4
)


# ============================================================
# 16. BOTONES RÁPIDOS
# ============================================================

ax_one = fig.add_axes(
    [0.065, 0.07, 0.08, 0.055]
)

ax_century = fig.add_axes(
    [0.160, 0.07, 0.08, 0.055]
)

ax_10k = fig.add_axes(
    [0.255, 0.07, 0.08, 0.055]
)

ax_100k = fig.add_axes(
    [0.350, 0.07, 0.08, 0.055]
)

ax_1m = fig.add_axes(
    [0.445, 0.07, 0.08, 0.055]
)

ax_play = fig.add_axes(
    [0.560, 0.07, 0.10, 0.055]
)

ax_reset = fig.add_axes(
    [0.675, 0.07, 0.10, 0.055]
)


button_one = Button(
    ax_one,
    "1 órbita"
)

button_century = Button(
    ax_century,
    "1 siglo"
)

button_10k = Button(
    ax_10k,
    "10 mil"
)

button_100k = Button(
    ax_100k,
    "100 mil"
)

button_1m = Button(
    ax_1m,
    "1 millón"
)

button_play = Button(
    ax_play,
    "Reproducir"
)

button_reset = Button(
    ax_reset,
    "Reiniciar"
)


# ============================================================
# 17. ACTUALIZACIÓN DE LA SIMULACIÓN
# ============================================================

def update_simulation(revolutions):
    """
    Actualiza la orientación de la órbita relativista,
    la línea del perihelio, la posición del planeta y
    los datos mostrados.
    """

    N = float(revolutions)

    # --------------------------------------------------------
    # Precesión acumulada
    # --------------------------------------------------------

    omega = accumulated_precession(N)


    # --------------------------------------------------------
    # Rotación de la órbita relativista
    # --------------------------------------------------------

    x_rot, y_rot = rotate_orbit(
        omega
    )

    gr_line.set_data(
        x_rot,
        y_rot
    )


    # --------------------------------------------------------
    # Dirección actual del perihelio
    # --------------------------------------------------------

    x_peri = (
        r_perihelion_AU
        * np.cos(omega)
    )

    y_peri = (
        r_perihelion_AU
        * np.sin(omega)
    )

    current_apsis_line.set_data(
        [0, x_peri],
        [0, y_peri]
    )


    # --------------------------------------------------------
    # Posición de Mercurio
    # --------------------------------------------------------
    #
    # La parte entera de N cuenta revoluciones completas.
    # La parte decimal determina en qué punto de la
    # revolución actual se encuentra Mercurio.

    fraction = N % 1

    radial_phase = (
        2 * np.pi * fraction
    )

    r_mercury = (
        p
        /
        (
            1
            + e * np.cos(radial_phase)
        )
    )

    phi_mercury = (
        radial_phase
        + omega
    )

    x_mercury = (
        r_mercury
        * np.cos(phi_mercury)
        / AU
    )

    y_mercury = (
        r_mercury
        * np.sin(phi_mercury)
        / AU
    )

    mercury.set_data(
        [x_mercury],
        [y_mercury]
    )


    # --------------------------------------------------------
    # Información mostrada
    # --------------------------------------------------------

    years = equivalent_years(N)

    omega_arcsec = (
        omega
        * RAD_TO_ARCSEC
    )

    omega_degrees = (
        np.degrees(omega)
    )

    info = (
        f"Revoluciones: {N:,.1f}\n"
        f"Tiempo equivalente: {years:,.1f} años\n"
        f"Precesión acumulada: {omega_arcsec:,.2f}''\n"
        f"Ángulo acumulado: {omega_degrees:.4f}°\n"
        f"Δφ por órbita: {delta_arcsec_orbit:.6f}''"
    )

    info_text.set_text(
        info
    )

    fig.canvas.draw_idle()


# Estado inicial de la simulación.

update_simulation(
    revolution_slider.val
)


# Cuando se mueve el slider, actualizamos todo.

revolution_slider.on_changed(
    update_simulation
)


# ============================================================
# 18. BOTONES RÁPIDOS
# ============================================================

def set_one_orbit(event):
    revolution_slider.set_val(
        1
    )


def set_one_century(event):
    revolution_slider.set_val(
        ORBITS_PER_CENTURY
    )


def set_10k(event):
    revolution_slider.set_val(
        10_000
    )


def set_100k(event):
    revolution_slider.set_val(
        100_000
    )


def set_1m(event):
    revolution_slider.set_val(
        1_000_000
    )


button_one.on_clicked(
    set_one_orbit
)

button_century.on_clicked(
    set_one_century
)

button_10k.on_clicked(
    set_10k
)

button_100k.on_clicked(
    set_100k
)

button_1m.on_clicked(
    set_1m
)


# ============================================================
# 19. MOSTRAR / OCULTAR NEWTON
# ============================================================

def toggle_newton(label):
    """
    Permite ocultar la órbita newtoniana cuando se quiera
    concentrar la atención únicamente en la relativista.
    """

    newton_line.set_visible(
        not newton_line.get_visible()
    )

    fig.canvas.draw_idle()


check_newton.on_clicked(
    toggle_newton
)


# ============================================================
# 20. ANIMACIÓN AUTOMÁTICA
# ============================================================

animation_running = False

# Este número controla únicamente la rapidez visual.
# No modifica epsilon, Delta_phi ni ningún parámetro físico.
#
# La parte decimal hace que Mercurio no aparezca siempre
# exactamente en el mismo punto de su órbita.

ANIMATION_STEP = 2000.37


def animation_frame(frame):
    global animation_running

    if not animation_running:
        return

    current_value = (
        revolution_slider.val
    )

    new_value = (
        current_value
        + ANIMATION_STEP
    )

    if new_value >= revolution_slider.valmax:

        new_value = (
            revolution_slider.valmax
        )

        animation_running = False

        button_play.label.set_text(
            "Reproducir"
        )

    revolution_slider.set_val(
        new_value
    )


animation = FuncAnimation(
    fig,
    animation_frame,
    interval=50,
    cache_frame_data=False
)


def toggle_animation(event):
    global animation_running

    animation_running = (
        not animation_running
    )

    if animation_running:

        button_play.label.set_text(
            "Pausar"
        )

    else:

        button_play.label.set_text(
            "Reproducir"
        )


button_play.on_clicked(
    toggle_animation
)


# ============================================================
# 21. REINICIO
# ============================================================

def reset_simulation(event):
    global animation_running

    animation_running = False

    button_play.label.set_text(
        "Reproducir"
    )

    # Reiniciamos en un siglo porque es la escala
    # característica del resultado clásico de Mercurio.

    revolution_slider.set_val(
        ORBITS_PER_CENTURY
    )


button_reset.on_clicked(
    reset_simulation
)


# ============================================================
# 22. MOSTRAR SIMULACIÓN
# ============================================================

plt.show()