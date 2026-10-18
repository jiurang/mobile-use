"""Android deployment surface for mobile-use.

Running the agent on the phone itself has one hard requirement that nothing in
the normal desktop setup cares about: the device layer speaks ADB, and there is
no ADB to speak with. See ``minitap.mobile_use.android.server`` for the HTTP
surface and the launch scripts in this package's directory for the Termux
wiring.
"""

from __future__ import annotations
