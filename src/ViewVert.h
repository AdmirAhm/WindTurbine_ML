//
//  Created by Izudin Dzafic on 28/07/2020.
//  Copyright © 2020 IDz. All rights reserved.
//
#pragma once
#include <gui/View.h>
#include <gui/VerticalLayout.h>
#include <gui/Label.h>
#include <gui/LineEdit.h>
#include <gui/TextEdit.h>
#include <gui/Button.h>

class ViewVert : public gui::View
{
private:
protected:
    gui::NumericEdit _t;
    gui::Label _lblt;
    gui::NumericEdit _w;
    gui::Label _lblw;
    gui::NumericEdit _beta;
    gui::Label _lblbeta;
    gui::GridLayout _gl;
    gui::GridComposer gc;

public:
    ViewVert()
    : _t(td::real8)
    , _w(td::real8)
    , _beta(td::real8)
    , _lblt("Vrijeme [s]")
    , _lblw("Brzina [rad/s]")
    , _lblbeta("Pitch [deg]")
    , _gl(3, 2)
    , gc(_gl)
    {
        gc.appendRow(_lblt);
        gc.appendCol(_t);
        gc.appendRow(_lblw);
        gc.appendCol(_w);
        gc.appendRow(_lblbeta);
        gc.appendCol(_beta);
        setLayout(&_gl);
    }
    
    void setValues(double t, double w, double beta) {
        _t.setValue(t);
        _w.setValue(w);
        _beta.setValue(beta);
    }
};
