const Gio = imports.gi.Gio;
const Shell = imports.gi.Shell;

const BUS_NAME = 'org.pllm.Foreground';
const OBJECT_PATH = '/org/pllm/Foreground';
const INTERFACE_XML = `
<node>
  <interface name="org.pllm.Foreground">
    <method name="GetActive">
      <arg type="u" direction="out" name="pid"/>
      <arg type="s" direction="out" name="app_id"/>
      <arg type="s" direction="out" name="title"/>
      <arg type="s" direction="out" name="wm_class"/>
    </method>
  </interface>
</node>`;

class ForegroundService {
    constructor() {
        this.pid = 0;
        this.appId = '';
        this.title = '';
        this.wmClass = '';
    }

    GetActive() {
        return [this.pid, this.appId, this.title, this.wmClass];
    }
}

class Extension {
    enable() {
        this._service = new ForegroundService();
        this._exported = Gio.DBusExportedObject.wrapJSObject(
            INTERFACE_XML,
            this._service
        );
        this._exported.export(Gio.DBus.session, OBJECT_PATH);
        this._ownerId = Gio.bus_own_name(
            Gio.BusType.SESSION,
            BUS_NAME,
            Gio.BusNameOwnerFlags.NONE,
            null,
            null,
            null
        );
        this._focusSignal = global.display.connect(
            'notify::focus-window',
            () => this._update()
        );
        this._update();
    }

    disable() {
        if (this._focusSignal) {
            global.display.disconnect(this._focusSignal);
            this._focusSignal = 0;
        }
        if (this._ownerId) {
            Gio.bus_unown_name(this._ownerId);
            this._ownerId = 0;
        }
        if (this._exported) {
            this._exported.unexport();
            this._exported = null;
        }
        this._service = null;
    }

    _update() {
        const window = global.display.get_focus_window();
        if (!window) {
            this._service.pid = 0;
            this._service.appId = '';
            this._service.title = '';
            this._service.wmClass = '';
            return;
        }
        const tracker = Shell.WindowTracker.get_default();
        const app = tracker.get_window_app(window);
        this._service.pid = Math.max(0, window.get_pid());
        this._service.appId = app ? app.get_id() : '';
        this._service.title = window.get_title() || '';
        this._service.wmClass = window.get_wm_class() || '';
    }
}

function init() {
    return new Extension();
}
