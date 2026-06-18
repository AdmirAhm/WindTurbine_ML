//
//  Created by Izudin Dzafic on 28/07/2020.
//  Copyright © 2020 IDz. All rights reserved.
//
#pragma once
#include <gui/SplitterLayout.h>
#include <gui/View.h>
#include <gui/Label.h>
#include <gui/NumericEdit.h>
#include <gui/CheckBox.h>
#include <gui/ColorPicker.h>
#include <gui/Button.h>
#include <gui/GridLayout.h>
#include <gui/GridComposer.h>
#include <gui/Timer.h>
#include "ViewGLLighting.h"
#include <gui//HorizontalLayout.h>
#include "ViewVert.h"


const float FPS = 10.0f; //desired frame per second
const float dT = 1/FPS;


class MainView : public gui::View
{
protected:
    gui::SplitterLayout _splitter;
    gui::Timer _timer;
    std::function<void()>* _pUpdateMenuAndTB;
    std::function<void()> _fnSetVisualEffects;
    ViewGLLighting _view;
    ViewVert _view2;
    gui::HorizontalLayout _hl;
protected:
    
public:
    MainView()
    : _splitter(gui::SplitterLayout::Orientation::Horizontal, gui::SplitterLayout::AuxiliaryCell::Second)
    , _timer(this, dT, false)
    , _hl(1)
    {
        setMargins(0, 0, 0, 0);
        _splitter.setContent(_view, _view2);
        setLayout(&_splitter);
        if (_timer.isRunning())
        {
            _timer.stop();
        }
    }
    
    bool isRunning() const
    {
        return _timer.isRunning();
    }
    
    void startStop()
    {
        if (_timer.isRunning())
        {
            _timer.stop();
            _view.stop();
        }
        else
        {
            _view.start();
            _timer.start();
        }
    }

    bool onTimer(gui::Timer* pTimer) override
    {
        double t, w, beta;
        _view.runStep(t, w, beta);
        _view2.setValues(t, w, beta);
        return true;
    }
    
};





